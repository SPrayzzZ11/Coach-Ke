#!/usr/bin/env python3
"""将 EPUB 的文档章节物理转换为独立 Markdown 文件。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import ITEM_DOCUMENT, epub


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


BLOCK_CONTAINERS = {"body", "div", "section", "article", "main", "header", "footer"}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def clean_text(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def inline_text(tag: Tag) -> str:
    for br in tag.find_all("br"):
        br.replace_with(NavigableString("\n"))
    return clean_text(tag.get_text(" ", strip=True))


def render_list(tag: Tag, ordered: bool) -> list[str]:
    lines: list[str] = []
    items = tag.find_all("li", recursive=False)
    for index, item in enumerate(items, 1):
        marker = f"{index}." if ordered else "-"
        text = inline_text(item)
        if text:
            lines.append(f"{marker} {text}")
    return lines


def render_table(tag: Tag) -> list[str]:
    rows: list[list[str]] = []
    for row in tag.find_all("tr"):
        cells = [inline_text(cell) for cell in row.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(cells)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return lines


def render_blocks(container: Tag) -> list[str]:
    lines: list[str] = []
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in {"script", "style", "noscript"}:
            continue
        if name in HEADINGS:
            text = inline_text(child)
            if text:
                lines.append(f"{'#' * HEADINGS[name]} {text}")
            continue
        if name == "p":
            text = inline_text(child)
            if text:
                lines.append(text)
            continue
        if name == "blockquote":
            for line in render_blocks(child):
                lines.append(f"> {line}")
            continue
        if name in {"ul", "ol"}:
            lines.extend(render_list(child, ordered=name == "ol"))
            continue
        if name == "pre":
            text = child.get_text("\n", strip=False).strip()
            if text:
                lines.extend(["```", text, "```"])
            continue
        if name == "table":
            lines.extend(render_table(child))
            continue
        if name in BLOCK_CONTAINERS:
            lines.extend(render_blocks(child))
            continue
        text = inline_text(child)
        if text:
            lines.append(text)
    return [line for line in lines if line.strip()]


def chapter_title(soup: BeautifulSoup, fallback: str) -> str:
    for heading in soup.find_all(list(HEADINGS)):
        title = inline_text(heading)
        if title:
            return title
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        title = inline_text(title_tag)
        if title:
            return title
    return fallback


def safe_filename(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:100] or "untitled"


def ordered_documents(book: epub.EpubBook) -> list[epub.EpubHtml]:
    documents = {
        item.get_id(): item
        for item in book.get_items_of_type(ITEM_DOCUMENT)
        if isinstance(item, epub.EpubHtml)
    }
    ordered: list[epub.EpubHtml] = []
    seen: set[str] = set()
    for spine_entry in book.spine:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = documents.get(str(item_id))
        if item and item.get_id() not in seen:
            ordered.append(item)
            seen.add(item.get_id())
    ordered.extend(item for item_id, item in documents.items() if item_id not in seen)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", type=Path, help="EPUB 文件路径")
    parser.add_argument("--output", type=Path, required=True, help="Markdown 输出目录")
    args = parser.parse_args()

    if not args.epub.exists():
        print(f"找不到 EPUB：{args.epub}", file=sys.stderr)
        return 1

    book = epub.read_epub(str(args.epub), options={"ignore_ncx": True})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for index, item in enumerate(ordered_documents(book), 1):
        html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for unwanted in soup.find_all(["script", "style", "noscript"]):
            unwanted.decompose()
        title = chapter_title(soup, item.get_name())
        body = soup.body or soup
        blocks = render_blocks(body)
        if not blocks:
            continue

        stem = f"{index:04d}_{safe_filename(title)}"
        output_path = args.output / f"{stem}.md"
        suffix = 2
        while output_path.exists():
            output_path = args.output / f"{stem}_{suffix}.md"
            suffix += 1
        header = [
            f"# {title}",
            "",
            f"<!-- epub_item: {item.get_name()} -->",
            "",
        ]
        output_path.write_text("\n".join(header + blocks).rstrip() + "\n", encoding="utf-8")
        manifest.append(
            {
                "order": index,
                "title": title,
                "epub_item": item.get_name(),
                "markdown": output_path.name,
            }
        )

    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "source_epub": str(args.epub),
                "documents": manifest,
                "document_count": len(manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已提取 {len(manifest)} 个 Markdown 文档到：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
