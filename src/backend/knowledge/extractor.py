"""
知识提取 & 知识图谱构建模块
使用 LLM 从章节中提取知识点 + 关系，组装为图谱 JSON
"""

import os, re, json
from typing import Optional
from llm_client import call_llm

# ── Prompt 模板 ───────────────────────────────

SYS_EXTRACT = """你是学科知识提取专家。从教材章节中提取核心知识点。
严格要求：
1. 输出纯 JSON 数组，不含其他文字
2. 每个知识点: name, definition, category, chapter
3. category ∈ {核心概念, 定理/定律, 方法/技术, 现象/过程, 结构/组成}
4. 每章节提取 3-8 个最重要知识点"""

PROMPT_EXTRACT = """教材：{book}
章节：{chapter}
内容：
{content}
输出 JSON 数组：
[{{"name":"...","definition":"...","category":"核心概念","chapter":"{chapter}"}}]"""

SYS_RELATION = """你是知识图谱关系识别专家。识别知识点间关系。
严格要求：
1. 输出纯 JSON 数组
2. relation_type ∈ {prerequisite, parallel, contains, applies_to}
3. 只输出确实存在的关系"""

PROMPT_RELATION = """知识点列表：
{points}
输出 JSON：
[{{"source":"A名称","target":"B名称","relation_type":"prerequisite","description":"一句话描述"}}]"""

# ── 公共 API ──────────────────────────────────

def build_extraction_prompt(textbook_name: str, chapter_title: str, content: str) -> str:
    return PROMPT_EXTRACT.format(book=textbook_name, chapter=chapter_title, content=content[:3000])


def extract_knowledge_nodes(textbook_name: str, chapter_title: str, content: str) -> list:
    """从章节提取知识点节点"""
    prompt = build_extraction_prompt(textbook_name, chapter_title, content)
    try:
        raw = call_llm(prompt, SYS_EXTRACT, temperature=0.3)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"[extract] LLM error: {e}")
    return _fallback_extract(chapter_title, content)


def _fallback_extract(chapter_title: str, content: str) -> list:
    sentences = re.split(r'[。；]', content)
    nodes = []
    for i, s in enumerate(sentences[:5]):
        s = s.strip()
        if len(s) > 10:
            nodes.append({"name": s[:30], "definition": s[:100],
                          "category": "核心概念", "chapter": chapter_title})
    return nodes


def extract_relations(nodes: list) -> list:
    """识别知识点间关系"""
    if len(nodes) < 2:
        return []
    summary = "\n".join(f"- {n.get('name','')}: {n.get('definition','')[:60]}" for n in nodes)
    prompt = PROMPT_RELATION.format(points=summary)
    try:
        raw = call_llm(prompt, SYS_RELATION, temperature=0.3)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"[relation] LLM error: {e}")
    return []


def build_knowledge_graph(book: dict) -> dict:
    """为单本教材构建完整知识图谱"""
    nodes, edges = [], []
    name_to_id = {}
    counter = 0
    for ch in book.get("chapters", []):
        ch_nodes = extract_knowledge_nodes(
            book.get("title", ""), ch.get("title", ""), ch.get("content", "")
        )
        for nd in ch_nodes:
            counter += 1
            uid = f"{book.get('textbook_id','b')}_n{counter:04d}"
            nd["id"] = uid
            nd["chapter"] = ch.get("title", "")
            nd["page"] = ch.get("page_start", 1)
            nd["textbook"] = book.get("title", "")
            nd["textbook_id"] = book.get("textbook_id", "")
            name_to_id[nd.get("name", "").strip()] = uid
            nodes.append(nd)
    for rel in extract_relations(nodes):
        s, t = name_to_id.get(rel.get("source", "").strip()), name_to_id.get(rel.get("target", "").strip())
        if s and t:
            edges.append({"source": s, "target": t,
                          "relation_type": rel.get("relation_type", "parallel"),
                          "description": rel.get("description", "")})
    return {
        "textbook_id": book.get("textbook_id", ""),
        "title": book.get("title", ""),
        "nodes": nodes, "edges": edges,
        "stats": {"total_nodes": len(nodes), "total_edges": len(edges)}
    }


def merge_graphs(graphs: list) -> dict:
    """合并多个图谱（用于后续整合）"""
    all_nodes, all_edges = [], []
    for g in graphs:
        all_nodes.extend(g.get("nodes", []))
        all_edges.extend(g.get("edges", []))
    return {"nodes": all_nodes, "edges": all_edges}


# ── ECharts 格式 ─────────────────────────────

_CAT_COLOR = {"核心概念": "#5470c6", "定理/定律": "#91cc75",
              "方法/技术": "#fac858", "现象/过程": "#ee6666", "结构/组成": "#73c0de"}

def graph_to_echarts(graph: dict) -> dict:
    nodes = []
    for n in graph.get("nodes", [])[:200]:  # 限制节点数
        nodes.append({
            "id": n.get("id", ""), "name": n.get("name", ""),
            "category": n.get("category", "核心概念"),
            "symbolSize": max(20, min(60, n.get("frequency", 1) * 15 + 15)),
            "itemStyle": {"color": _CAT_COLOR.get(n.get("category", ""), "#999")},
        })
    links = []
    for e in graph.get("edges", [])[:300]:
        links.append({
            "source": e.get("source", ""), "target": e.get("target", ""),
            "label": {"show": True, "formatter": e.get("relation_type", "")}
        })
    return {
        "nodes": nodes, "links": links,
        "categories": [{"name": k} for k in _CAT_COLOR]
    }

