"""
RAG Pipeline: Chunk → Embed → Index → Retrieve → Generate
"""

import os, re, json
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    textbook_name: str = ""
    chapter_title: str = ""
    page: int = 1


@dataclass
class Citation:
    textbook: str
    chapter: str
    page: int
    relevance_score: float
    chunk_text: str = ""


@dataclass
class RAGResponse:
    answer: str
    citations: list = field(default_factory=list)
    source_chunks: list = field(default_factory=list)

    def to_dict(self):
        return {
            "answer": self.answer,
            "citations": [
                {"textbook": c.textbook, "chapter": c.chapter,
                 "page": c.page, "relevance_score": round(c.relevance_score, 4)}
                for c in self.citations
            ],
            "source_chunks": self.source_chunks,
        }


# ── Chunking ──────────────────────────────────

class TextChunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_textbook(self, book: dict) -> list:
        """将一本教材拆分为 chunk 列表"""
        chunks = []
        for ch in book.get("chapters", []):
            text = ch.get("content", "")
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunk_id = f"{book.get('textbook_id','b')}_{ch.get('chapter_id','c')}_{start}"
                chunks.append(Chunk(
                    id=chunk_id, text=chunk_text,
                    textbook_name=book.get("title", ""),
                    chapter_title=ch.get("title", ""),
                    page=ch.get("page_start", 1),
                ))
                start += self.chunk_size - self.overlap
        return chunks


# ── Embedding & Index ─────────────────────────

class VectorIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings = None
        self._model = None

    def _get_embed_fn(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
            self._model = SentenceTransformer(model_name)
            return self._model
        except ImportError:
            return None

    def index(self, chunks: list[Chunk]) -> str:
        """建立向量索引"""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        model = self._get_embed_fn()
        if model:
            self.embeddings = model.encode(texts, normalize_embeddings=True)
            return f"已索引 {len(chunks)} 个知识块（sentence-transformers）"
        # Fallback: 使用 numpy 零向量占位
        import numpy as np
        self.embeddings = np.random.randn(len(texts), 384).astype('float32')
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        return f"已索引 {len(chunks)} 个知识块（fallback 模式）"

    def search(self, query: str, top_k: int = 5) -> list:
        """检索最相关的 top_k 个 chunk"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        import numpy as np
        model = self._get_embed_fn()
        if model:
            q_emb = model.encode([query], normalize_embeddings=True)[0]
        else:
            q_emb = np.random.randn(384).astype('float32')
            q_emb /= np.linalg.norm(q_emb)
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        results = []
        for idx in top_idx:
            c = self.chunks[idx]
            results.append(Citation(
                textbook=c.textbook_name, chapter=c.chapter_title,
                page=c.page, relevance_score=float(sims[idx]),
                chunk_text=c.text,
            ))
        return results


# ── Generate ──────────────────────────────────

from llm_client import call_llm


RAG_SYSTEM = """你是学科教学助手。只基于提供的教材内容回答问题。
规则：
1. 只使用上下文中的信息，不编造
2. 每个回答附带引用来源：[教材名, 章节, 页码]
3. 如果上下文中找不到答案，回复"当前知识库中未找到相关信息"
4. 回答简洁准确"""


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n---\n\n".join(
        f"[来源: {c.textbook_name}, {c.chapter_title}, p{c.page}]\n{c.text}"
        for c in chunks
    )
    return f"""请基于以下教材内容回答问题。

## 参考资料
{ctx}

## 问题
{question}

请给出答案并附上引用来源。"""


def rag_query(question: str, vector_index: VectorIndex, top_k: int = 5) -> RAGResponse:
    """执行一次 RAG 查询"""
    citations = vector_index.search(question, top_k)
    if not citations:
        return RAGResponse(answer="当前知识库中未找到相关信息")

    chunks_for_prompt = [
        Chunk(id="", text=c.chunk_text, textbook_name=c.textbook,
              chapter_title=c.chapter, page=c.page)
        for c in citations
    ]
    prompt = build_rag_prompt(question, chunks_for_prompt)
    answer = call_llm(prompt, RAG_SYSTEM, temperature=0.3)

    return RAGResponse(
        answer=answer,
        citations=citations,
        source_chunks=[c.chunk_text for c in citations],
    )

