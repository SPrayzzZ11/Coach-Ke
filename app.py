#!/usr/bin/env python3
"""柯教 Web 聊天界面。"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
BUILD_INDEX_SCRIPT = ROOT / "scripts" / "build_index.py"
sys.path.insert(0, str(ROOT / "scripts"))

from retrieve import (  # noqa: E402
    COLLECTION_NAME,
    MODEL_NAME,
    RETRIEVAL_DEPTH,
    bm25_search,
    expand_query,
    load_bm25,
    reciprocal_rank_fusion,
    vector_search,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
COVERAGE_THRESHOLD = 0.02
MAX_HISTORY_TURNS = 5
NO_COVERAGE_REPLY = "我是柯教。当前知识库未覆盖此问题，请补充相关资料后再来问我。"
SYSTEM_PROMPT = (
    "你现在是‘柯教’，一名基于《力量训练基础》体系的严谨力量举教练。"
    "你需要结合提供的知识块，给出专业、直接、可执行的建议。"
    "绝对不要废话，不要脱离知识块自行发挥。"
    "如果涉及伤病，必须以教练身份提醒安全就医。"
)


@st.cache_resource(show_spinner=False)
def ensure_knowledge_index() -> None:
    """在全新部署环境中构建一次本地检索索引。"""
    if BM25_INDEX_PATH.exists():
        return

    completed = subprocess.run(
        [sys.executable, str(BUILD_INDEX_SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not BM25_INDEX_PATH.exists():
        details = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        raise RuntimeError(
            "知识库索引自动构建失败。"
            + (f"\n\n{details[-4000:]}" if details else "")
        )


@dataclass(frozen=True)
class SearchHit:
    id: str
    source_book: str
    topic: str
    content_summary: str
    source_locator: str
    rrf_score: float


@dataclass(frozen=True)
class RetrievalEngine:
    model: SentenceTransformer | None
    collection: object | None
    bm25_index: dict
    metadata_by_id: dict[str, dict]
    documents_by_id: dict[str, str]
    vector_error: str | None


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    used_vector: bool
    warning: str | None


@st.cache_resource(show_spinner=False)
def load_retrieval_engine() -> RetrievalEngine:
    """加载一次向量模型与两类本地索引，供同一 Web 会话复用。"""
    bm25_path = (DATA_DIR / "bm25_index.pkl").resolve()
    chroma_path = (DATA_DIR / "chroma_db").resolve()
    if not bm25_path.exists():
        raise FileNotFoundError(
            "本地 BM25 索引不存在，请先运行 "
            r".\.venv\Scripts\python.exe scripts\build_index.py"
        )

    bm25_index = load_bm25(bm25_path)
    metadata_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(bm25_index["ids"], bm25_index["metadatas"])
    }
    documents_by_id = dict(zip(bm25_index["ids"], bm25_index["documents"]))
    model: SentenceTransformer | None = None
    collection = None
    vector_error: str | None = None
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() == 0:
            raise RuntimeError(f"集合 {COLLECTION_NAME} 为空")
        try:
            model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        vector_error = f"向量检索不可用，已回退到 BM25：{exc}"
        collection = None
    return RetrievalEngine(
        model=model,
        collection=collection,
        bm25_index=bm25_index,
        metadata_by_id=metadata_by_id,
        documents_by_id=documents_by_id,
        vector_error=vector_error,
    )


def retrieve_top_hits(query: str, top: int = 3) -> RetrievalResult:
    """复用 CLI 检索脚本的查询扩展、双路召回与 RRF 融合。"""
    engine = load_retrieval_engine()
    expanded_query = expand_query(query)
    vector_ids: list[str] = []
    warning = engine.vector_error
    used_vector = False
    if engine.model is not None and engine.collection is not None:
        try:
            vector_ids, _ = vector_search(
                expanded_query,
                engine.model,
                engine.collection,
                RETRIEVAL_DEPTH,
            )
            used_vector = True
        except Exception as exc:
            warning = f"向量检索不可用，已回退到 BM25：{exc}"
    bm25_ids, _ = bm25_search(
        expanded_query,
        engine.bm25_index,
        RETRIEVAL_DEPTH,
    )
    fused_scores = reciprocal_rank_fusion(vector_ids, bm25_ids)
    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )[:top]

    hits: list[SearchHit] = []
    for chunk_id in ranked_ids:
        metadata = engine.metadata_by_id.get(chunk_id, {})
        hits.append(
            SearchHit(
                id=chunk_id,
                source_book=str(metadata.get("source_book", "")),
                topic=str(metadata.get("topic", "")),
                content_summary=engine.documents_by_id.get(chunk_id, ""),
                source_locator=str(metadata.get("source_locator", "")),
                rrf_score=float(fused_scores[chunk_id]),
            )
        )
    return RetrievalResult(hits=hits, used_vector=used_vector, warning=warning)


def build_grounded_system_prompt(query: str, hits: list[SearchHit]) -> str:
    blocks = []
    for rank, hit in enumerate(hits, 1):
        blocks.append(
            "\n".join(
                (
                    f"[知识块 {rank}]",
                    f"id: {hit.id}",
                    f"source_book: {hit.source_book}",
                    f"topic: {hit.topic}",
                    f"source_locator: {hit.source_locator}",
                    f"content_summary: {hit.content_summary}",
                )
            )
        )
    return (
        SYSTEM_PROMPT
        + "\n\n请只依据以下已审核知识块回答当前问题，并综合互补信息。"
        "历史对话只用于理解指代和上下文，不得把历史回答当作新事实根据。"
        "不要补充知识块之外的动作、剂量或诊断。\n\n"
        "[本轮知识块]\n"
        + "\n\n".join(blocks)
        + f"\n\n[本轮用户问题]\n{query}"
    )


def recent_history(messages: list[dict]) -> list[dict]:
    conversation = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    return conversation[-(MAX_HISTORY_TURNS * 2):]


def stream_deepseek_answer(
    api_key: str,
    query: str,
    hits: list[SearchHit],
    history: list[dict],
):
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=60.0,
    )
    messages = [
        {
            "role": "system",
            "content": build_grounded_system_prompt(query, hits),
        },
        *recent_history(history),
        {"role": "user", "content": query},
    ]
    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def render_sources(hits: list[SearchHit]) -> None:
    if not hits:
        return
    with st.expander("👀 柯教的备课笔记 (知识溯源)"):
        for rank, hit in enumerate(hits, 1):
            st.markdown(f"**{rank}. {hit.topic}**")
            st.caption(f"`{hit.id}` · RRF `{hit.rrf_score:.6f}`")
            st.write(hit.content_summary)
            if hit.source_locator:
                st.caption(f"来源位置：{hit.source_locator}")
            if rank != len(hits):
                st.divider()


def render_message(message: dict) -> None:
    role = message["role"]
    with st.chat_message(role, avatar="🏋️" if role == "assistant" else "🧑"):
        st.markdown(message["content"])
    render_sources(message.get("sources", []))


st.set_page_config(
    page_title="柯教 - AI专属私人教练",
    page_icon="🏋️",
    layout="centered",
)

if not BM25_INDEX_PATH.exists():
    st.info("首次部署，正在自动构建知识库索引，请稍候...")
try:
    ensure_knowledge_index()
except Exception as exc:
    st.error(f"知识库初始化失败：{exc}")
    st.stop()

st.title("🏋️ 柯教 (Coach Ke)")
st.caption("以《力量训练基础》为知识底座的力量训练问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("柯教")
    st.write(
        "我是柯教，你的专属力量教练。"
        "我的大脑目前收录了《力量训练基础》的三大项"
        "（深蹲、卧推、推举）硬核知识。"
    )
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        help="密钥仅用于本次浏览器会话调用 DeepSeek，不会写入仓库。",
    )
    st.caption("知识块在本地检索；提问与 Top 3 知识块会发送给 DeepSeek 生成回答。")
    if st.button(
        "清空对话",
        icon=":material/delete:",
        width="stretch",
        key="clear_chat",
    ):
        st.session_state.messages = []
        st.rerun()

for saved_message in st.session_state.messages:
    render_message(saved_message)

prompt = st.chat_input(
    "向柯教提问，例如：卧推时肩膀疼怎么办？",
    submit_mode="disable",
)
if prompt:
    history = recent_history(st.session_state.messages)
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        retrieval = retrieve_top_hits(prompt, top=3)
        hits = retrieval.hits
    except Exception as exc:
        response_text = f"柯教加载本地检索索引失败：{exc}"
        st.error(response_text)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "sources": [],
            }
        )
        st.rerun()
    else:
        if retrieval.warning:
            st.warning(retrieval.warning)
        response_text: str | None = None
        with st.chat_message("assistant", avatar="🏋️"):
            coverage_threshold = COVERAGE_THRESHOLD if retrieval.used_vector else 0.01
            if not hits or hits[0].rrf_score < coverage_threshold:
                response_text = NO_COVERAGE_REPLY
                st.markdown(response_text)
            elif not api_key.strip():
                response_text = "请先在侧边栏填写 DeepSeek API Key，柯教才能组织回答。"
                st.info(response_text)
            else:
                try:
                    streamed = st.write_stream(
                        stream_deepseek_answer(
                            api_key.strip(),
                            prompt,
                            hits,
                            history,
                        )
                    )
                    response_text = streamed if isinstance(streamed, str) else str(streamed)
                except Exception as exc:
                    response_text = (
                        "柯教暂时无法连接 DeepSeek。"
                        f"请检查 API Key、网络和账户额度后重试。错误信息：{exc}"
                    )
                    st.error(response_text)

        render_sources(hits)
        if response_text:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "sources": hits,
                }
            )
            st.rerun()
