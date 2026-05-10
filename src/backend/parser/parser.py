"""
多格式教材文件解析模块
支持: PDF, Markdown, TXT, DOCX
输出统一结构化 JSON / dict
"""

import os, re, json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Chapter:
    chapter_id: str
    title: str
    level: int = 1
    page_start: int = 1
    page_end: int = 1
    content: str = ""
    char_count: int = 0

    def __post_init__(self):
        if self.content and not self.char_count:
            self.char_count = len(self.content)


@dataclass
class TextbookInfo:
    textbook_id: str
    filename: str
    title: str
    format: str = "unknown"
    total_pages: int = 0
    total_chars: int = 0
    chapters: list = field(default_factory=list)
    status: str = "pending"

    def to_dict(self):
        return {
            "textbook_id": self.textbook_id,
            "filename": self.filename,
            "title": self.title,
            "format": self.format,
            "total_pages": self.total_pages,
            "total_chars": self.total_chars,
            "chapters": [asdict(c) for c in self.chapters],
            "status": self.status,
        }


# ── 章节模式 ──────────────────────────────────
CHAPTER_PATTERNS = [
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+篇\s'), 1),
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+章\s'), 2),
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+节\s'), 3),
    (re.compile(r'^\s*\d+\.\d+\.\d+\s'), 4),
    (re.compile(r'^\s*\d+\.\d+\s'), 3),
    (re.compile(r'^\s*\d+[\.\、]\s'), 2),
    (re.compile(r'^\s*[\[（(][一二三四五六七八九十\d]+[\]）)]\s*'), 2),
]

def _detect_level(text: str) -> int:
    for pat, lv in CHAPTER_PATTERNS:
        if pat.match(text):
            return lv
    return 0


# ── 各格式解析器 ──────────────────────────────

def parse_pdf(filepath: str, textbook_id: str) -> TextbookInfo:
    import fitz
    doc = fitz.open(filepath)
    path = Path(filepath)
    book = TextbookInfo(
        textbook_id=textbook_id,
        filename=path.name,
        title=path.stem,
        format="pdf",
        total_pages=len(doc),
        status="parsing",
    )
    chapters = []
    cur = None
    counter = 0
    for pg in range(len(doc)):
        text = doc[pg].get_text("text")
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            lv = _detect_level(line)
            if lv >= 1 and lv <= 3 and len(line) < 80:
                if cur:
                    cur.page_end = pg + 1
                    cur.char_count = len(cur.content)
                    chapters.append(cur)
                counter += 1
                cur = Chapter(
                    chapter_id=f"ch_{counter:02d}",
                    title=line, level=lv,
                    page_start=pg + 1, page_end=pg + 1,
                )
            elif cur:
                cur.content += line + "\n"
    if cur:
        cur.page_end = book.total_pages
        cur.char_count = len(cur.content)
        chapters.append(cur)
    doc.close()
    book.chapters = chapters
    book.total_chars = sum(c.char_count for c in chapters)
    book.status = "done"
    return book


def parse_text(filepath: str, textbook_id: str) -> TextbookInfo:
    """解析 Markdown / TXT"""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    fmt = "markdown" if ext == ".md" else "txt"
    book = TextbookInfo(
        textbook_id=textbook_id,
        filename=path.name,
        title=path.stem,
        format=fmt,
        total_pages=1,
        status="parsing",
    )
    chapters = []
    cur = None
    counter = 0
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('#'):
            lv = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            if lv <= 3:
                if cur:
                    cur.char_count = len(cur.content)
                    chapters.append(cur)
                counter += 1
                cur = Chapter(
                    chapter_id=f"ch_{counter:02d}",
                    title=title, level=lv,
                )
        elif cur:
            cur.content += line + "\n"
    if cur:
        cur.char_count = len(cur.content)
        chapters.append(cur)
    if not chapters:
        chapters.append(Chapter(
            chapter_id="ch_01", title=path.stem, level=1,
            content=text,
        ))
    book.chapters = chapters
    book.total_chars = sum(c.char_count for c in chapters)
    book.status = "done"
    return book


def parse_docx(filepath: str, textbook_id: str) -> TextbookInfo:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("需要 python-docx: pip install python-docx")
    path = Path(filepath)
    doc = Document(filepath)
    full = "\n".join(p.text for p in doc.paragraphs)
    book = TextbookInfo(
        textbook_id=textbook_id,
        filename=path.name,
        title=path.stem,
        format="docx",
        total_pages=1,
        total_chars=len(full),
        status="done",
    )
    book.chapters = [Chapter(
        chapter_id="ch_01", title=path.stem, level=1,
        content=full,
    )]
    return book


PARSERS = {
    "pdf": parse_pdf,
    "md": parse_text,
    "markdown": parse_text,
    "txt": parse_text,
    "docx": parse_docx,
}

def parse_textbook(filepath: str, textbook_id: str) -> TextbookInfo:
    ext = Path(filepath).suffix.lower().lstrip('.')
    fn = PARSERS.get(ext)
    if not fn:
        raise ValueError(f"不支持格式: .{ext}，支持: {list(PARSERS.keys())}")
    return fn(filepath, textbook_id)

