#!/usr/bin/env python3
"""使用 ChromaDB 向量检索与 BM25 进行 RRF 混合检索。"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "fitness_chunks"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RRF_K = 60
RETRIEVAL_DEPTH = 5
SYNONYMS = {
    "起桥": ["起桥", "胸部高挺", "上背稳定", "肩胛骨内收", "背部收紧", "腿部支撑"],
    "肩膀疼": ["肩膀疼", "肩部夹挤", "肘部外展", "肩部耸动"],
    "腰酸": ["腰酸", "腰椎过度伸展", "骨盆前倾", "腹部收紧"],
    "头晕": ["头晕", "呼吸", "大脑缺氧"],
}
STOPWORDS = (
    "怎么办",
    "怎么回事",
    "怎么走",
    "如何",
    "为什么",
    "什么",
    "请问",
    "时候",
    "总是",
    "这个",
    "那个",
    "一下",
    "我的",
    "我",
    "你",
    "吗",
    "呢",
    "的",
    "了",
    "时",
    "在",
    "是",
    "和",
    "与",
    "或",
    "要",
    "有",
)
ALIASES = {
    "轨迹": ("路径", "方向"),
    "杠铃轨迹": ("杠铃路径",),
    "上背松": ("上背", "上背松弛", "上背失去紧绷", "紧绷"),
    "腰疼": ("腰椎", "腰椎过度伸展", "腰部"),
    "腰酸": ("腰椎", "腰椎过度伸展", "腰部", "躯干控制"),
    "腰": ("腰椎", "腰部"),
    "头晕": ("短暂头晕", "晕厥"),
}

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def expand_query(query: str) -> str:
    """把命中的口语化关键词扩展为垂直领域术语。"""
    expansions: list[str] = []
    for trigger, terms in SYNONYMS.items():
        if trigger not in query:
            continue
        for term in terms:
            if term not in query and term not in expansions:
                expansions.append(term)
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"


def tokenize(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", value).lower()
    for stopword in STOPWORDS:
        text = text.replace(stopword, " ")
    for source, aliases in ALIASES.items():
        if source in text:
            text += " " + " ".join(aliases)

    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9]+", segment):
            tokens.append(segment)
            continue
        if 2 <= len(segment) <= 8:
            tokens.append(segment)
        for size in (2, 3, 4):
            if len(segment) >= size:
                tokens.extend(segment[index:index + size] for index in range(len(segment) - size + 1))
    return tokens


def load_bm25(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到 BM25 索引：{path}；请先运行 scripts/build_index.py")
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict) or "bm25" not in data or "ids" not in data:
        raise ValueError(f"BM25 索引格式无效：{path}")
    return data


def vector_search(
    query: str,
    model: SentenceTransformer,
    collection,
    depth: int,
) -> tuple[list[str], dict[str, float]]:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    count = collection.count()
    if count == 0:
        return [], {}
    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(depth, count),
        include=["distances", "documents", "metadatas"],
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return list(ids), {
        chunk_id: float(distance)
        for chunk_id, distance in zip(ids, distances)
    }


def bm25_search(query: str, index: dict, depth: int) -> tuple[list[str], dict[str, float]]:
    query_tokens = tokenize(query)
    raw_scores = index["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(raw_scores)),
        key=lambda index_no: (-float(raw_scores[index_no]), index["ids"][index_no]),
    )
    ranked_indices = [index_no for index_no in ranked_indices if raw_scores[index_no] > 0][:depth]
    ids = [index["ids"][index_no] for index_no in ranked_indices]
    return ids, {
        index["ids"][index_no]: float(raw_scores[index_no])
        for index_no in ranked_indices
    }


def reciprocal_rank_fusion(
    vector_ids: list[str],
    bm25_ids: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids in (vector_ids, bm25_ids):
        for rank, chunk_id in enumerate(ranked_ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--top", type=int, default=3, help="最终返回数量，默认 3")
    parser.add_argument(
        "--chroma",
        type=Path,
        default=DATA_DIR / "chroma_db",
        help="ChromaDB 持久化目录",
    )
    parser.add_argument(
        "--bm25",
        type=Path,
        default=DATA_DIR / "bm25_index.pkl",
        help="BM25 pickle 索引路径",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Sentence Transformers 模型名",
    )
    args = parser.parse_args()

    if args.top < 1:
        print("--top 必须大于 0", file=sys.stderr)
        return 2

    try:
        bm25_index = load_bm25(args.bm25.expanduser().resolve())
    except Exception as exc:
        print(f"加载 BM25 索引失败：{exc}", file=sys.stderr)
        return 1

    expanded_query = expand_query(args.query)
    vector_ids: list[str] = []
    vector_scores: dict[str, float] = {}
    chroma_path = args.chroma.expanduser().resolve()
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
            model = SentenceTransformer(args.model, local_files_only=True)
        except Exception:
            model = SentenceTransformer(args.model)
        vector_ids, vector_scores = vector_search(
            expanded_query,
            model,
            collection,
            RETRIEVAL_DEPTH,
        )
    except Exception as exc:
        print(
            f"ChromaDB 不可用，已回退到 BM25 检索：{exc}",
            file=sys.stderr,
        )

    bm25_ids, bm25_scores = bm25_search(expanded_query, bm25_index, RETRIEVAL_DEPTH)
    fused_scores = reciprocal_rank_fusion(vector_ids, bm25_ids)

    metadata_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(bm25_index["ids"], bm25_index["metadatas"])
    }
    documents_by_id = dict(zip(bm25_index["ids"], bm25_index["documents"]))
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, 1)}
    bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, 1)}

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )[:args.top]
    if not ranked_ids:
        print("无匹配的已审核知识块。")
        return 0

    for rank, chunk_id in enumerate(ranked_ids, 1):
        metadata = metadata_by_id.get(chunk_id, {})
        print(f"[{rank}] rrf_score={fused_scores[chunk_id]:.6f}")
        print(f"id: {chunk_id}")
        print(f"topic: {metadata.get('topic', '')}")
        print(f"content_summary: {documents_by_id.get(chunk_id, '')}")
        print(f"vector_rank: {vector_rank.get(chunk_id, '-')}")
        print(f"bm25_rank: {bm25_rank.get(chunk_id, '-')}")
        if chunk_id in vector_scores:
            print(f"vector_distance: {vector_scores[chunk_id]:.6f}")
        if chunk_id in bm25_scores:
            print(f"bm25_score: {bm25_scores[chunk_id]:.6f}")
        if rank != len(ranked_ids):
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
