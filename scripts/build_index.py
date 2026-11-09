#!/usr/bin/env python3
"""构建 ChromaDB 向量索引和本地 BM25 索引。"""

from __future__ import annotations

import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "fitness_chunks"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
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


def load_reviewed_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {line_no} 行 JSON 解析失败：{exc.msg}") from exc
        if chunk.get("review_status") == "reviewed" and chunk.get("needs_review") is False:
            chunks.append(chunk)
    return chunks


def main() -> int:
    chunks_path = (PROJECT_ROOT / "knowledge" / "chunks.jsonl").resolve()
    chroma_path = (DATA_DIR / "chroma_db").resolve()
    bm25_path = (DATA_DIR / "bm25_index.pkl").resolve()

    if not chunks_path.exists():
        print(f"找不到知识块文件：{chunks_path}", file=sys.stderr)
        return 1
    chunks = load_reviewed_chunks(chunks_path)
    if not chunks:
        print("没有 review_status 为 reviewed 的知识块，无法构建索引。", file=sys.stderr)
        return 1

    ids = [str(chunk["id"]) for chunk in chunks]
    documents = [str(chunk["content_summary"]) for chunk in chunks]
    metadatas = [
        {
            "source_book": str(chunk.get("source_book", "")),
            "topic": str(chunk.get("topic", "")),
            "chunk_type": str(chunk.get("chunk_type", "")),
            "source_locator": str(chunk.get("source_locator", "")),
        }
        for chunk in chunks
    ]

    print(f"加载向量模型：{MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        print("本地模型缓存不可用，尝试从 Hugging Face 下载。")
        model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    tokenized_corpus = [tokenize(document) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as file:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "tokenized_corpus": tokenized_corpus,
                "bm25": bm25,
            },
            file,
        )

    print(f"已构建向量索引：{len(ids)} 条 -> {chroma_path}")
    print(f"已构建 BM25 索引：{len(ids)} 条 -> {bm25_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
