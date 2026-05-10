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

# 子模式: 仅在已有章节内触发子章节（第X节、多级编号）
# 注意: 以下模式被刻意排除，因为它们是正文编号列表而非章节标题：
#   - \d+[\.\、]\s   → "1. 定义" 是列表项，非章节
#   - （一）（二）    → 正文子项，非章节
SECONDARY_PATTERNS = [
    (re.compile(r'^\s*第[一二三四五六七八九十百千\d]+节\s'), 3),
    (re.compile(r'^\s*\d+\.\d+\.\d+\s'), 4),
    (re.compile(r'^\s*\d+\.\d+\s'), 3),
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

def parse_pdf(filepath: str, textbook_id: str, toc_max_level: int = 1) -> TextbookInfo:
    """
    解析 PDF 教材
    
    Args:
        toc_max_level: TOC 提取的最大层级。1=仅章, 2=章+节, 3=章+节+子节。
                       设得越小，章节越少（与真实教材目录一致）。
    """
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

    # ── 策略: 优先使用 PDF 内置大纲 ──
    toc = doc.get_toc()
    # 需要 >= 5 个 Lv1 条目才认为 TOC 可靠（排除只有封面/封底的假 TOC）
    l1_count = sum(1 for item in toc if item[0] == 1)
    if l1_count >= 5:
        chapters = _parse_pdf_with_toc(doc, toc, book, toc_max_level)
    else:
        chapters = _parse_pdf_regex(doc, book)

    book.chapters = chapters
    # total_chars 去重：对每页只计一次字数，避免 TOC 父子层级重叠导致重复计算
    unique_chars = 0
    seen_pages = set()
    for ch in chapters:
        for pg in range(ch.page_start, ch.page_end + 1):
            if pg not in seen_pages:
                seen_pages.add(pg)
                if 0 <= pg - 1 < book.total_pages:
                    page_text = doc[pg - 1].get_text("text")
                    unique_chars += len(page_text)
    book.total_chars = unique_chars
    book.status = "done"
    doc.close()
    return book


# ── 非内容页面标题关键词（这些页面的内容不应作为章节） ──
_NON_CONTENT_TITLES = {
    '封面页', '封面', '书名页', '版权页', '版权', '编委名单', '编委',
    '新形态教材使用说明', '使用说明', '序言', '教材修订说明',
    '主审简介', '主编简介', '副主编简介', '前言', '目录',
    '推荐阅读', '中英文名词对照索引', '索引', '封底页', '封底',
}


def _is_content_chapter(title: str) -> bool:
    """判断 TOC 条目是否为正文章节（排除封面/目录/索引等）"""
    title_clean = title.strip().replace('\u3000', '').replace(' ', '')
    for kw in _NON_CONTENT_TITLES:
        if kw in title_clean:
            return False
    return True


def _parse_pdf_with_toc(doc, toc: list, book: TextbookInfo, max_level: int = 1) -> list:
    """
    基于 PDF 内置大纲 (TOC) 解析章节结构。
    max_level=1 时只保留章级别条目（如"第一章 绪论"），
    max_level=2 保留章+节，max_level=3 保留全部层级。
    自动去重：只保留最深可达层级的条目，父条目页面被剥离给子条目。
    """
    chapters = []
    counter = 0
    total_pages = len(doc)

    # ── 步骤 1: TOC 转条目列表 ──
    toc_entries = []
    for level, title, page in toc:
        toc_entries.append({
            'level': level,
            'title': title.strip(),
            'page': max(1, min(page, total_pages)),
        })

    # 计算 page_end：遇到同级或更高级条目时截止
    for i, entry in enumerate(toc_entries):
        entry['page_end'] = total_pages
        for j in range(i + 1, len(toc_entries)):
            if toc_entries[j]['level'] <= entry['level']:
                entry['page_end'] = max(entry['page'], toc_entries[j]['page'] - 1)
                break

    # ── 步骤 2: 消除重叠 —— Lv3/Lv2 同级条目间移除页面重叠 ──
    for lv in [3, 2]:
        prev_entry = None
        for entry in toc_entries:
            if entry['level'] == lv:
                if prev_entry is not None and entry['page'] <= prev_entry['page_end']:
                    entry['page'] = prev_entry['page_end'] + 1
                if entry['page'] > entry['page_end']:
                    entry['page'] = entry['page_end']
                prev_entry = entry
            elif entry['level'] < lv:
                prev_entry = None

    # ── 步骤 3: 找到第一个正文章节 ──
    first_content_idx = None
    for i, entry in enumerate(toc_entries):
        if entry['level'] == 1 and _is_content_chapter(entry['title']):
            first_content_idx = i
            break

    # ── 步骤 4: 前言 ──
    preface_start = 1
    preface_end = toc_entries[first_content_idx]['page'] - 1 if first_content_idx else total_pages
    if preface_start <= preface_end:
        preface_text = _extract_page_range(doc, preface_start, preface_end)
        if preface_text.strip():
            counter += 1
            chapters.append(Chapter(
                chapter_id=f"ch_{counter:02d}",
                title="前言 / 序言",
                level=0,
                page_start=preface_start,
                page_end=preface_end,
                content=preface_text,
            ))

    if first_content_idx is None:
        return chapters

    # ── 步骤 5: 按 TOC 创建章节（仅保留 max_level 以内的层级） ──
    for entry in toc_entries[first_content_idx:]:
        if not _is_content_chapter(entry['title']):
            continue
        if entry['level'] > max_level:
            continue  # 跳过超出指定深度的子条目，其页面归入父条目
        if entry['page'] > entry['page_end']:
            continue

        text = _extract_page_range(doc, entry['page'], entry['page_end'])
        counter += 1
        chapters.append(Chapter(
            chapter_id=f"ch_{counter:02d}",
            title=entry['title'],
            level=entry['level'],
            page_start=entry['page'],
            page_end=entry['page_end'],
            content=text,
        ))

    return chapters


def _extract_page_range(doc, start_page: int, end_page: int) -> str:
    """提取指定页码范围（1-based）的文本，过滤页眉页脚"""
    if start_page > end_page:
        return ""
    text_parts = []
    for pg in range(start_page - 1, min(end_page, len(doc))):
        page_text = doc[pg].get_text("text")
        for line in page_text.split('\n'):
            stripped = line.strip()
            if not _is_header_footer(stripped):
                text_parts.append(stripped)
    return "\n".join(text_parts)


def _parse_pdf_regex(doc, book: TextbookInfo) -> list:
    """回退方案: 基于正则逐行扫描解析章节（保留原有逻辑）"""
    chapters = []
    cur = None
    counter = 0
    preface_content = ""
    preface_start_page = 1
    found_first_primary = False
    recent_primary_titles = {}

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
    return _merge_duplicate_chapters(chapters)


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

def parse_textbook(filepath: str, textbook_id: str, **kwargs) -> TextbookInfo:
    ext = Path(filepath).suffix.lower().lstrip('.')
    fn = PARSERS.get(ext)
    if not fn:
        raise ValueError(f"不支持格式: .{ext}，支持: {list(PARSERS.keys())}")
    return fn(filepath, textbook_id, **kwargs)


# ── Cache 管理 ──────────────────────────────────

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_to_cache(book: TextbookInfo) -> Path:
    """将解析结果保存到 ./cache/ 目录 (JSON 格式)"""
    cache_path = CACHE_DIR / f"{book.textbook_id}.json"
    cache_path.write_text(
        json.dumps(book.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_path


def load_from_cache(textbook_id: str) -> Optional[TextbookInfo]:
    """从 ./cache/ 加载已缓存的解析结果"""
    cache_path = CACHE_DIR / f"{textbook_id}.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    chapters = [
        Chapter(
            chapter_id=c["chapter_id"],
            title=c["title"],
            level=c.get("level", 1),
            page_start=c.get("page_start", 1),
            page_end=c.get("page_end", 1),
            content=c.get("content", ""),
            char_count=c.get("char_count", 0),
        )
        for c in data.get("chapters", [])
    ]
    return TextbookInfo(
        textbook_id=data["textbook_id"],
        filename=data["filename"],
        title=data["title"],
        format=data.get("format", "unknown"),
        total_pages=data.get("total_pages", 0),
        total_chars=data.get("total_chars", 0),
        chapters=chapters,
        status=data.get("status", "done"),
    )


def list_cache() -> list:
    """列出 ./cache/ 中所有缓存文件"""
    if not CACHE_DIR.exists():
        return []
    return sorted(
        [p.stem for p in CACHE_DIR.glob("*.json")],
        key=lambda x: (CACHE_DIR / f"{x}.json").stat().st_mtime,
        reverse=True,
    )


def clear_cache():
    """清空 ./cache/ 目录"""
    for p in CACHE_DIR.glob("*.json"):
        p.unlink()

