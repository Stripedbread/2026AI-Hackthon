"""
关键词检索模块 — 教师对话专用
流程: 提取关键词 → 关键词匹配搜索教材章节 → 返回相关片段
"""

import re
from collections import defaultdict
from llm_client import call_llm


# ── 关键词提取 Prompt ─────────────────────────

KEYWORD_EXTRACT_PROMPT = """从以下教师的问题中提取 3-5 个最关键的医学术语/概念名称。

规则：
1. 只提取医学术语、疾病名、生理过程名、药物名等专业概念
2. 去停用词（"是什么""为什么""怎么""的""了"等）
3. 每个关键词尽可能简短（2-6字）
4. 直接输出逗号分隔的关键词，不要解释

问题：{question}

关键词："""


def extract_keywords(question: str) -> list[str]:
    """从教师问题中提取关键词（LLM + 正则兜底）"""
    # 先尝试 LLM
    try:
        raw = call_llm(
            KEYWORD_EXTRACT_PROMPT.format(question=question),
            system="你是医学关键词提取专家。只输出逗号分隔的关键词，不输出其他内容。",
            temperature=0.1,
        )
        # 清理输出
        raw = raw.strip().rstrip("。，.。")
        if 2 <= len(raw) <= 60 and "[LLM" not in raw:
            keywords = [kw.strip() for kw in re.split(r'[,，、\s]+', raw) if 2 <= len(kw.strip()) <= 10]
            if keywords:
                return keywords[:5]
    except Exception:
        pass

    # 兜底：用正则从问题中提取可能的医学概念
    return _fallback_extract(question)


def _fallback_extract(question: str) -> list[str]:
    """正则兜底提取中文医学术语"""
    # 匹配《》引用的教材名或「」引用的概念
    concepts = []
    quoted = re.findall(r'[「《]([^》」]{2,20})[》」]', question)
    concepts.extend(quoted)

    # 匹配"XX的XX"模式（如"细胞的功能"）
    de_pattern = re.findall(r'([一-鿿]{2,8}的[一-鿿]{2,8})', question)
    concepts.extend(de_pattern)

    # 常见医学后缀词
    medical_suffix = r'(?:症|病|酶|素|细胞|组织|器官|系统|反应|作用|过程|机制|功能|调节|代谢|免疫|神经|血液|呼吸|消化|循环)'
    medical_terms = re.findall(rf'[一-鿿]{{2,4}}{medical_suffix}', question)
    concepts.extend(medical_terms)

    # 去重并限制数量
    seen = set()
    result = []
    for c in concepts:
        if c not in seen and len(c) >= 2:
            seen.add(c)
            result.append(c)
    return result[:5]


# ── 关键词搜索 ─────────────────────────────────

class KeywordRetriever:
    """基于关键词匹配的教材内容检索器"""

    def __init__(self, max_chars_per_result: int = 400, max_total_chars: int = 2000):
        self.max_chars_per_result = max_chars_per_result
        self.max_total_chars = max_total_chars
        self._book_cache: dict = {}  # book_id → parsed chapter data

    def index_books(self, books: dict):
        """缓存教材数据以便快速搜索"""
        self._book_cache = {}
        for bid, book in books.items():
            book_dict = book.to_dict() if hasattr(book, "to_dict") else book
            chapters = book_dict.get("chapters", [])
            self._book_cache[bid] = {
                "title": book_dict.get("title", ""),
                "filename": book_dict.get("filename", ""),
                "chapters": chapters,
            }

    def search(self, keywords: list[str], top_n: int = 5) -> list[dict]:
        """搜索教材内容，返回匹配度最高的片段

        Returns:
            [{textbook, chapter, page, content, score, matched_keywords}, ...]
        """
        if not keywords or not self._book_cache:
            return []

        # 为每个章节打分
        scored = []
        for bid, book_data in self._book_cache.items():
            for ch in book_data["chapters"]:
                content = ch.get("content", "")
                if len(content) < 20:
                    continue
                score, matched = self._score_chapter(content, keywords)
                if score > 0:
                    scored.append({
                        "textbook": book_data["title"],
                        "chapter": ch.get("title", ""),
                        "page": ch.get("page_start", 1),
                        "content": self._extract_best_snippet(content, keywords),
                        "score": score,
                        "matched_keywords": matched,
                    })

        # 按分数降序排列
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    def _score_chapter(self, content: str, keywords: list[str]) -> tuple:
        """计算章节与关键词的匹配分数"""
        score = 0
        matched = []
        content_lower = content.lower()
        for kw in keywords:
            count = content_lower.count(kw.lower())
            if count > 0:
                # 每个关键词匹配 + 出现次数加权
                kw_score = 1.0 + min(count * 0.2, 2.0)
                score += kw_score
                matched.append(kw)
        return score, matched

    def _extract_best_snippet(self, content: str, keywords: list[str]) -> str:
        """从章节内容中提取包含关键词最多的片段"""
        if len(content) <= self.max_chars_per_result:
            return content

        # 滑动窗口找到关键词密度最高的区域
        window = self.max_chars_per_result
        step = window // 2
        best_start = 0
        best_score = 0
        content_lower = content.lower()

        for start in range(0, len(content) - window + 1, step):
            snippet = content_lower[start:start + window]
            score = sum(snippet.count(kw.lower()) for kw in keywords)
            if score > best_score:
                best_score = score
                best_start = start

        snippet = content[best_start:best_start + window]
        # 在词边界截断
        if best_start > 0:
            snippet = "…" + snippet
        if best_start + window < len(content):
            snippet = snippet + "…"
        return snippet

    def format_results(self, results: list[dict]) -> str:
        """将检索结果格式化为 LLM 可读文本"""
        if not results:
            return ""

        total_chars = 0
        lines = ["## 🔍 关键词检索结果（以下为教材中匹配的原文片段）\n"]
        for i, r in enumerate(results, 1):
            content = r["content"]
            # 控制总长度
            if total_chars + len(content) > self.max_total_chars:
                content = content[:self.max_total_chars - total_chars] + "…"
            lines.append(
                f"**[来源{i}]** 《{r['textbook']}》{r['chapter']} 第{r['page']}页 "
                f"(匹配: {', '.join(r['matched_keywords'])})\n"
                f"> {content}\n"
            )
            total_chars += len(content)
            if total_chars >= self.max_total_chars:
                break

        return "\n".join(lines)

    def retrieve_and_format(self, question: str, top_n: int = 5) -> str:
        """一站式：提取关键词 → 搜索 → 格式化"""
        keywords = extract_keywords(question)
        if not keywords:
            return ""
        results = self.search(keywords, top_n)
        if not results:
            return ""
        formatted = self.format_results(results)
        return formatted
