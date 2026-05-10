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
# 主模式: 始终触发新章节（第X篇、第X章）
PRIMARY_PATTERNS = [
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+篇\s'), 1),
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+章\s'), 2),
]

# 子模式: 仅在已有章节内触发子章节（第X节、编号、括号编号）
SECONDARY_PATTERNS = [
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+节\s'), 3),
    (re.compile(r'^\s*\d+\.\d+\.\d+\s'), 4),
    (re.compile(r'^\s*\d+\.\d+\s'), 3),
    (re.compile(r'^\s*\d+[\.\、]\s'), 2),
    (re.compile(r'^\s*[\[（(][一二三四五六七八九十\d]+[\]）)]\s*'), 2),
]

# 页眉/页脚过滤: 纯数字、罗马数字、短于4字符的行
HEADER_FOOTER_RE = re.compile(
    r'^\s*(\d{1,4}|[IVXivx]+|[A-Z]-\d+)\s*$'
)

def _is_header_footer(line: str) -> bool:
    """判断是否为页眉或页脚"""
    line = line.strip()
    if not line:
        return True
    if len(line) <= 3:
        return True
    if HEADER_FOOTER_RE.match(line):
        return True
    return False

def _detect_level_primary(text: str):
    """检测主模式，返回 (level, is_primary)"""
    for pat, lv in PRIMARY_PATTERNS:
        if pat.match(text):
            return lv, True
    for pat, lv in SECONDARY_PATTERNS:
        if pat.match(text):
            return lv, False
    return 0, False

def _detect_level(text: str) -> int:
    """兼容旧接口"""
    lv, _ = _detect_level_primary(text)
    return lv


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
    preface_content = ""       # 第一个主章节前的所有内容
    preface_start_page = 1
    found_first_primary = False
    recent_primary_titles = {}  # title → chapter index (用于去重页眉)

    for pg in range(len(doc)):
        text = doc[pg].get_text("text")
        lines = text.split('\n')

        # ── TOC 检测: 一页中有 >= 4 条主模式行 → 目录页，全部归前言 ──
        primary_matches_on_page = sum(
            1 for line in lines
            if _detect_level_primary(line.strip())[1]
            and len(line.strip()) < 80
        )
        is_toc_page = primary_matches_on_page >= 4

        for line in lines:
            stripped = line.strip()
            if _is_header_footer(stripped):
                continue

            lv, is_primary = _detect_level_primary(stripped)

            # ── TOC 页面: 全部内容归前言，不创建章节 ──
            if is_toc_page:
                preface_content += stripped + "\n"
                continue

            # ── 主模式: 第X篇 / 第X章 ──
            if is_primary and lv >= 1 and lv <= 2 and len(stripped) < 80:
                # 标准化标题（统一 Unicode 空白字符，防止"第十章 "和"第十章 "被视为不同）
                normalized_title = re.sub(r'[\s\u3000\u00A0\u2000-\u200F\u2028-\u202F]+', ' ', stripped).strip()
                # 运行页眉去重: 同名主章节已存在 → 跳过
                if normalized_title in recent_primary_titles:
                    continue

                if cur:
                    cur.page_end = pg + 1
                    cur.char_count = len(cur.content)
                    chapters.append(cur)

                # 如果之前有前言内容，先保存
                if not found_first_primary and preface_content.strip():
                    counter += 1
                    chapters.append(Chapter(
                        chapter_id=f"ch_{counter:02d}",
                        title="前言 / 序言",
                        level=0,
                        page_start=preface_start_page,
                        page_end=pg,
                        content=preface_content,
                    ))
                found_first_primary = True
                counter += 1
                cur = Chapter(
                    chapter_id=f"ch_{counter:02d}",
                    title=stripped, level=lv,
                    page_start=pg + 1, page_end=pg + 1,
                )
                recent_primary_titles[normalized_title] = len(chapters) - 1

            # ── 子模式: 第X节 / 编号 → 仅在已有章节内 ──
            elif not is_primary and lv >= 1 and lv <= 3 and len(stripped) < 80:
                if not found_first_primary:
                    preface_content += stripped + "\n"
                    continue
                if cur:
                    cur.page_end = pg + 1
                    cur.char_count = len(cur.content)
                    chapters.append(cur)
                counter += 1
                cur = Chapter(
                    chapter_id=f"ch_{counter:02d}",
                    title=stripped, level=lv,
                    page_start=pg + 1, page_end=pg + 1,
                )
            elif cur:
                cur.content += stripped + "\n"
            elif not found_first_primary:
                if not preface_content:
                    preface_start_page = pg + 1
                preface_content += stripped + "\n"

    # 保存最后一个章节
    if cur:
        cur.page_end = book.total_pages
        cur.char_count = len(cur.content)
        chapters.append(cur)

    # 如果全书都没有主章节，将全部内容作为前言
    if not found_first_primary and preface_content.strip():
        chapters.insert(0, Chapter(
            chapter_id="ch_00",
            title="前言 / 全书内容",
            level=0,
            page_start=1,
            page_end=book.total_pages,
            content=preface_content,
        ))

    # ── 后处理: 合并连续同名章节 ──
    chapters = _merge_duplicate_chapters(chapters)

    doc.close()
    book.chapters = chapters
    book.total_chars = sum(c.char_count for c in chapters)
    book.status = "done"
    return book


def _merge_duplicate_chapters(chapters: list) -> list:
    """合并连续同名章节（处理极少量的 TOC 边缘情况）"""
    if len(chapters) <= 1:
        return chapters
    merged = []
    i = 0
    while i < len(chapters):
        cur_ch = chapters[i]
        j = i + 1
        while j < len(chapters) and chapters[j].title == cur_ch.title:
            if chapters[j].content:
                cur_ch.content += chapters[j].content
            cur_ch.page_end = max(cur_ch.page_end, chapters[j].page_end)
            cur_ch.char_count = len(cur_ch.content)
            j += 1
        merged.append(cur_ch)
        i = j
    for idx, ch in enumerate(merged):
        ch.chapter_id = f"ch_{idx + 1:02d}"
    return merged


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

