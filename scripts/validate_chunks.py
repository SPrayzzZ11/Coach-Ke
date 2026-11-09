#!/usr/bin/env python3
"""校验健身示例块与正式知识块的结构、状态和引用关系。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


REQUIRED_FIELDS = (
    "id",
    "source_book",
    "topic",
    "chunk_type",
    "applicable_to",
    "priority",
    "content_summary",
    "source_locator",
    "sample",
    "needs_review",
    "review_status",
    "reviewed_by",
    "last_reviewed_at",
    "supersedes",
    "superseded_by",
    "deprecated_reason",
)

CHUNK_TYPES = {
    "动作要点",
    "原则",
    "常见错误",
    "安全提示",
    "计划变量",
    "营养规则",
}
APPLICABLE_TO = {"新手", "进阶", "通用"}
PRIORITIES = {"high", "medium", "low"}
EXAMPLE_STATUSES = {"sample"}
FORMAL_STATUSES = {"draft", "reviewed", "deprecated"}


def read_book_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*-\s+id:\s*["\']?([^"\'\s]+)', text, flags=re.MULTILINE))


def validate_chunk(
    chunk: object,
    line_no: int,
    book_ids: set[str],
    container_kind: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(chunk, dict):
        return [f"第 {line_no} 行不是 JSON 对象"]

    missing = [field for field in REQUIRED_FIELDS if field not in chunk]
    if missing:
        errors.append(f"第 {line_no} 行缺少字段：{', '.join(missing)}")

    if chunk.get("chunk_type") not in CHUNK_TYPES:
        errors.append(f"第 {line_no} 行 chunk_type 不合法")
    if chunk.get("applicable_to") not in APPLICABLE_TO:
        errors.append(f"第 {line_no} 行 applicable_to 不合法")
    if chunk.get("priority") not in PRIORITIES:
        errors.append(f"第 {line_no} 行 priority 不合法")
    if book_ids and chunk.get("source_book") not in book_ids:
        errors.append(f"第 {line_no} 行 source_book 不在 books.yaml 中")
    if not isinstance(chunk.get("sample"), bool):
        errors.append(f"第 {line_no} 行 sample 必须为布尔值")
    if not isinstance(chunk.get("needs_review"), bool):
        errors.append(f"第 {line_no} 行 needs_review 必须为布尔值")
    if not str(chunk.get("content_summary", "")).strip():
        errors.append(f"第 {line_no} 行 content_summary 不能为空")

    review_status = chunk.get("review_status")
    if container_kind == "examples":
        if review_status not in EXAMPLE_STATUSES:
            errors.append(f"第 {line_no} 行示例夹具只允许 review_status: sample")
        if chunk.get("sample") is not True:
            errors.append(f"第 {line_no} 行示例夹具必须 sample: true")
        if chunk.get("needs_review") is not True:
            errors.append(f"第 {line_no} 行示例夹具必须 needs_review: true")
        if chunk.get("reviewed_by") is not None or chunk.get("last_reviewed_at") is not None:
            errors.append(f"第 {line_no} 行示例夹具不得填写审核结果")
    else:
        if review_status not in FORMAL_STATUSES:
            errors.append(f"第 {line_no} 行正式知识块只允许 draft、reviewed 或 deprecated")
        if chunk.get("sample") is not False:
            errors.append(f"第 {line_no} 行正式知识块必须 sample: false")

    if review_status == "draft" and chunk.get("needs_review") is not True:
        errors.append(f"第 {line_no} 行 draft 状态必须 needs_review: true")
    if review_status in {"reviewed", "deprecated"} and chunk.get("needs_review") is not False:
        errors.append(f"第 {line_no} 行 {review_status} 状态必须 needs_review: false")
    if review_status == "reviewed":
        if not str(chunk.get("reviewed_by") or "").strip():
            errors.append(f"第 {line_no} 行 reviewed 状态必须填写 reviewed_by")
        reviewed_at = chunk.get("last_reviewed_at")
        if not reviewed_at:
            errors.append(f"第 {line_no} 行 reviewed 状态必须填写 last_reviewed_at")
        else:
            try:
                datetime.fromisoformat(str(reviewed_at).replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"第 {line_no} 行 last_reviewed_at 不是 ISO 日期时间")
    if review_status == "deprecated" and not str(chunk.get("deprecated_reason") or "").strip():
        errors.append(f"第 {line_no} 行 deprecated 状态必须填写 deprecated_reason")

    supersedes = chunk.get("supersedes")
    if not isinstance(supersedes, list) or any(not isinstance(item, str) for item in supersedes):
        errors.append(f"第 {line_no} 行 supersedes 必须为字符串数组")
    superseded_by = chunk.get("superseded_by")
    if superseded_by is not None and not isinstance(superseded_by, str):
        errors.append(f"第 {line_no} 行 superseded_by 必须为字符串或 null")
    deprecated_reason = chunk.get("deprecated_reason")
    if deprecated_reason is not None and not isinstance(deprecated_reason, str):
        errors.append(f"第 {line_no} 行 deprecated_reason 必须为字符串或 null")
    review_note = chunk.get("review_note")
    if review_note is not None and not isinstance(review_note, str):
        errors.append(f"第 {line_no} 行 review_note 必须为字符串或 null")
    return errors


def validate_file(
    path: Path,
    container_kind: str,
    expected_count: int | None,
    book_ids: set[str],
    seen_ids: set[str],
) -> tuple[list[str], int]:
    errors: list[str] = []
    if not path.exists():
        return [f"找不到知识块文件：{path}"], 0

    count = 0
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        count += 1
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name} 第 {line_no} 行 JSON 解析失败：{exc.msg}")
            continue

        errors.extend(validate_chunk(chunk, line_no, book_ids, container_kind))
        chunk_id = chunk.get("id") if isinstance(chunk, dict) else None
        if chunk_id in seen_ids:
            errors.append(f"{path.name} 第 {line_no} 行 id 重复：{chunk_id}")
        elif chunk_id:
            seen_ids.add(chunk_id)

    if expected_count is not None and count != expected_count:
        errors.append(f"{path.name} 数量为 {count}，预期为 {expected_count}")
    return errors, count


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="仅校验单个文件；默认同时校验 examples.jsonl 和 chunks.jsonl",
    )
    parser.add_argument("--expected-count", type=int, default=None, help="单文件模式下的预期数量")
    parser.add_argument("--expected-examples", type=int, default=None)
    parser.add_argument("--expected-chunks", type=int, default=None)
    args = parser.parse_args()

    books_path = root / "data" / "books.yaml"
    if not books_path.exists():
        print(f"找不到书单文件：{books_path}", file=sys.stderr)
        return 1
    book_ids = read_book_ids(books_path)

    if args.path:
        path = args.path
        kind = "examples" if path.name == "examples.jsonl" else "chunks"
        targets = [(path, kind, args.expected_count)]
    else:
        targets = [
            (root / "knowledge" / "examples.jsonl", "examples", args.expected_examples),
            (root / "knowledge" / "chunks.jsonl", "chunks", args.expected_chunks),
        ]

    errors: list[str] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    for path, kind, expected_count in targets:
        file_errors, count = validate_file(path, kind, expected_count, book_ids, seen_ids)
        errors.extend(file_errors)
        counts[kind] = counts.get(kind, 0) + count

    if errors:
        print("校验失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "校验通过："
        f"examples.jsonl {counts.get('examples', 0)} 条 sample 测试夹具；"
        f"chunks.jsonl {counts.get('chunks', 0)} 条正式知识块。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
