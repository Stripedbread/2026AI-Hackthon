"""
知识提取 & 知识图谱构建模块 (Enhanced for Competition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
严格按照赛题要求实现：
  1. 每个章节调用 LLM 提取核心知识点 (id, name, definition, category, chapter, page)
  2. 识别知识点间的 4 种关系：prerequisite, parallel, contains, applies_to
  3. 输出结构化 JSON 图谱数据
  4. 支持 ECharts 可视化格式（含教材来源颜色区分 & 频次大小）
"""

import os, re, json
from typing import Optional
from llm_client import call_llm

# ═══════════════════════════════════════════════
# Prompt 模板 — 知识点提取（含 Few-shot 示例）
# ═══════════════════════════════════════════════

SYS_EXTRACT = """你是学科知识图谱构建专家。你的任务是从教材章节中提取核心知识点，并输出严格的 JSON 格式结果。

## 知识点分类标准
- 核心概念：学科中的基础、关键概念（如"动作电位""内环境稳态"）
- 定理/定律：具有普遍性的规律或原理（如"Starling定律""全或无定律"）
- 方法/技术：研究或应用中使用的方法、技术（如"膜片钳技术""PCR"）
- 现象/过程：可观察的生理/病理现象或过程（如"炎症反应""血液凝固"）
- 结构/组成：器官、组织、细胞、分子的结构描述（如"细胞膜结构""肾单位"）

## 输出格式要求
- 必须输出纯 JSON 数组，外层不能有任何文字
- 每章提取 5-12 个最重要的知识点
- definition 字段要求 30-150 字的精炼定义
- 跳过目录、序言、索引等非教学内容

## Few-shot 示例
输入章节"细胞的基本功能-细胞膜的物质转运功能"：
```json
[
  {"name":"单纯扩散","definition":"脂溶性小分子物质顺浓度梯度直接穿过细胞膜的转运方式，不消耗能量，不需要膜蛋白协助，如O2、CO2的跨膜转运","category":"核心概念"},
  {"name":"易化扩散","definition":"水溶性或带电物质在膜转运蛋白的协助下顺浓度梯度或电化学梯度跨膜转运的过程，分为载体介导和通道介导两种","category":"核心概念"},
  {"name":"主动转运","definition":"细胞通过耗能过程将物质逆浓度梯度或电化学梯度进行跨膜转运，分为原发性主动转运和继发性主动转运","category":"核心概念"},
  {"name":"钠钾泵","definition":"即Na⁺-K⁺-ATP酶，每分解1分子ATP可将3个Na⁺泵出细胞、2个K⁺泵入细胞，是维持细胞内外Na⁺、K⁺浓度差的关键","category":"结构/组成"},
  {"name":"继发性主动转运","definition":"利用原发性主动转运建立的离子浓度梯度提供的势能，将另一种物质逆浓度梯度转运的过程，如葡萄糖-Na⁺同向转运","category":"核心概念"}
]
```"""

PROMPT_EXTRACT = """请从以下教材章节中提取核心知识点。

教材：{book}
章节：{chapter}
章节内容：
{content}

请严格按照上述格式输出 JSON 数组，每个知识点包含：
- name: 知识点名称（简短，5-20字）
- definition: 精确定义（30-150字）
- category: 分类（核心概念/定理·定律/方法·技术/现象·过程/结构·组成）
- chapter: "{chapter}"

只输出 JSON 数组，不使用代码块标记。"""


# ═══════════════════════════════════════════════
# Prompt 模板 — 关系识别
# ═══════════════════════════════════════════════

SYS_RELATION = """你是知识图谱关系识别专家。识别知识点间的逻辑关系。

## 关系类型定义
1. prerequisite（前置依赖）：学习B之前必须先掌握A，即A→B
   例："静息电位" → prerequisite → "动作电位"（理解动作电位必须先掌握静息电位）
2. parallel（并列关系）：同一层级、可互相参照的平行概念
   例："有丝分裂" ↔ parallel ↔ "减数分裂"
3. contains（包含关系）：上位概念包含下位概念
   例："免疫系统" → contains → "T细胞"
4. applies_to（应用关系）：某个知识点的应用场景或实践
   例："抗体" → applies_to → "体液免疫"

## 输出格式
- 纯 JSON 数组，只识别确实存在且明确的关系
- description 用一句话解释关系
- 每个知识点最多参与 3-5 个关系

## Few-shot 示例
```json
[
  {"source":"静息电位","target":"动作电位","relation_type":"prerequisite","description":"理解动作电位的产生机制必须先掌握静息电位的概念"},
  {"source":"细胞膜","target":"离子通道","relation_type":"contains","description":"离子通道是嵌入细胞膜的蛋白质结构"},
  {"source":"有丝分裂","target":"减数分裂","relation_type":"parallel","description":"两者都是细胞分裂方式，但机制和结果不同"}
]
```"""

PROMPT_RELATION = """以下是从教材中提取的知识点列表，请识别它们之间的关系。

知识点列表：
{points}

请输出 JSON 数组表示关系，每个关系包含 source, target, relation_type, description。
只输出确实存在的关系，不要为了凑数而编造关系。
只输出 JSON 数组，不使用代码块标记。"""


# ═══════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════

def _is_skip_chapter(title: str) -> bool:
    """判断是否为应跳过的非教学内容章节"""
    skip_keywords = [
        "前言", "序言", "目录", "索引", "参考文献", "编委", "编写说明",
        "出版说明", "修订说明", "主审简介", "主编简介", "副主编简介",
        "前言 / 序言", "推荐阅读", "中英文名词对照", "本章数字资源",
        "PREFACE", "Preface", "Foreword", "Contents"
    ]
    title_lower = title.lower().strip()
    for kw in skip_keywords:
        if kw.lower() in title_lower:
            return True
    # 跳过纯编号的碎片章节（如 "（一）", "1.", "一、" 等过短的标题）
    if len(title) < 5 and re.match(r'^[（(]?\d+[)）]?[.\s、]?$', title.strip()):
        return True
    return False


def _clean_json(raw: str) -> Optional[list]:
    """从 LLM 回复中提取 JSON 数组"""
    # 尝试直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # 提取 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 提取最外层 [ ... ]
    m = re.search(r'\[[\s\S]*\]', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # 提取 {...} 之间
    m = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _fallback_extract(chapter_title: str, content: str) -> list:
    """LLM 提取失败时的备用方案：使用规则提取"""
    nodes = []
    # 找关键术语（引号内、括号内、或定义句式）
    terms = set()
    # 引号中的术语
    for m in re.finditer(r'[「「]([^」」]+)[」」]', content):
        terms.add(m.group(1))
    for m in re.finditer(r'["""]([^"""]+)["""]', content):
        if len(m.group(1)) <= 20:
            terms.add(m.group(1))
    # 括号中的术语
    for m in re.finditer(r'（([^）]{2,30})）', content):
        terms.add(m.group(1))
    for m in re.finditer(r'\(([^)]{2,30})\)', content):
        if not re.search(r'[a-zA-Z]{3,}', m.group(1)):
            terms.add(m.group(1))
    # 定义句式：X是/是指/指/即
    for m in re.finditer(r'([\u4e00-\u9fff\w]{3,25})(?:是|是指|指|即|指的是|定义为)\s*([^。；，]{8,100})', content):
        name = m.group(1).strip()
        if len(name) >= 3 and len(name) <= 25:
            terms.add(name)
    
    for i, t in enumerate(list(terms)[:10]):
        nodes.append({
            "name": t[:30],
            "definition": f"教材中定义的概念：{t}"[:150],
            "category": "核心概念",
            "chapter": chapter_title,
        })
    if not nodes:
        sentences = re.split(r'[。；]', content)
        for i, s in enumerate(sentences[:5]):
            s = s.strip()
            if len(s) > 15:
                nodes.append({
                    "name": s[:30], "definition": s[:120],
                    "category": "核心概念", "chapter": chapter_title,
                })
    return nodes


# ═══════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════

def build_extraction_prompt(textbook_name: str, chapter_title: str, content: str) -> str:
    """构建提取 prompt（供外部调用）"""
    return PROMPT_EXTRACT.format(book=textbook_name, chapter=chapter_title,
                                  content=content[:5000])


def extract_knowledge_nodes(
    textbook_name: str, chapter_title: str, content: str, verbose: bool = True,
) -> list:
    """
    从章节内容中提取知识点节点（LLM 驱动）

    参数:
        textbook_name: 教材名称
        chapter_title: 章节标题
        content: 章节文本内容
        verbose: 是否打印进度

    返回: [
        {"name":"...", "definition":"...", "category":"...", "chapter":"..."},
        ...
    ]
    """
    max_len = 5000
    truncated = content[:max_len] if len(content) > max_len else content

    prompt = PROMPT_EXTRACT.format(book=textbook_name, chapter=chapter_title,
                                    content=truncated)

    try:
        raw = call_llm(prompt, SYS_EXTRACT, temperature=0.3)
        result = _clean_json(raw)
        if result and isinstance(result, list) and len(result) > 0:
            valid = []
            for item in result:
                if isinstance(item, dict) and "name" in item:
                    valid.append({
                        "name": str(item.get("name", ""))[:50],
                        "definition": str(item.get("definition", ""))[:200],
                        "category": str(item.get("category", "核心概念"))[:20],
                        "chapter": chapter_title,
                    })
            if len(valid) >= 2:
                if verbose:
                    print(f"  ✅ 提取 {len(valid)} 个知识点")
                return valid
    except Exception as e:
        if verbose:
            print(f"  ⚠️  LLM 提取失败: {e}")

    nodes = _fallback_extract(chapter_title, content)
    if verbose:
        print(f"  ⚠️  回退提取 {len(nodes)} 个知识点")
    return nodes


def extract_relations(nodes: list, verbose: bool = True) -> list:
    """
    识别知识点间关系（LLM 驱动）

    返回: [
        {"source":"名称A", "target":"名称B", "relation_type":"prerequisite", "description":"..."},
        ...
    ]
    """
    if len(nodes) < 2:
        return []

    lines = []
    for n in nodes:
        name = n.get("name", "")
        definition = n.get("definition", "")[:80]
        lines.append(f"- {name}: {definition}")
    summary = "\n".join(lines)

    prompt = PROMPT_RELATION.format(points=summary)

    try:
        raw = call_llm(prompt, SYS_RELATION, temperature=0.3)
        result = _clean_json(raw)
        if result and isinstance(result, list):
            valid = []
            valid_types = {"prerequisite", "parallel", "contains", "applies_to"}
            for item in result:
                if isinstance(item, dict) and "source" in item and "target" in item:
                    rt = item.get("relation_type", "parallel")
                    if rt not in valid_types:
                        rt = "parallel"
                    valid.append({
                        "source": str(item.get("source", "")).strip(),
                        "target": str(item.get("target", "")).strip(),
                        "relation_type": rt,
                        "description": str(item.get("description", ""))[:120],
                    })
            if verbose:
                print(f"  ✅ 识别 {len(valid)} 个关系")
            return valid
    except Exception as e:
        if verbose:
            print(f"  ⚠️  关系识别失败: {e}")

    return []


# ═══════════════════════════════════════════════
# 图谱构建
# ═══════════════════════════════════════════════

def build_knowledge_graph(
    book: dict,
    progress_callback=None,
    verbose: bool = True,
    max_chapters: int = 50,
) -> dict:
    """
    为单本教材构建完整知识图谱（赛题 P0 要求）

    Args:
        book: 教材 dict
        progress_callback: 进度回调
        verbose: 是否打印详细信息
        max_chapters: 每本教材最多处理的章节数（限制 LLM 调用次数）

    返回格式:
    {
        "textbook_id", "title",
        "nodes": [{"id","name","definition","category","chapter","page","textbook","textbook_id"}],
        "edges": [{"source","target","relation_type","description"}],
        "stats": {"total_nodes":N, "total_edges":M}
    }
    """
    chapters = book.get("chapters", [])
    valid_chapters = [
        ch for ch in chapters
        if not _is_skip_chapter(ch.get("title", ""))
        and ch.get("char_count", 0) >= 500
    ]
    
    # 限制章节数（取内容最长的章节优先）
    if len(valid_chapters) > max_chapters:
        valid_chapters.sort(key=lambda c: c.get("char_count", 0), reverse=True)
        valid_chapters = valid_chapters[:max_chapters]
        # 按页码恢复排序
        valid_chapters.sort(key=lambda c: c.get("page_start", 0))

    if verbose:
        print(f"\n{'='*60}")
        print(f"📘 构建知识图谱: {book.get('title','')}")
        print(f"   有效章节: {len(valid_chapters)}/{len(chapters)}")
        print(f"{'='*60}")

    nodes, edges = [], []
    name_to_id = {}
    counter = 0

    total = len(valid_chapters)
    for ci, ch in enumerate(valid_chapters):
        ch_title = ch.get("title", "")
        ch_content = ch.get("content", "")
        page_start = ch.get("page_start", 1)

        if progress_callback:
            progress_callback(
                (ci + 0.1) / total,
                f"[{ci+1}/{total}] 提取: {ch_title[:30]}"
            )

        if verbose:
            print(f"\n  [{ci+1}/{total}] 📖 {ch_title}")

        ch_nodes = extract_knowledge_nodes(
            book.get("title", ""), ch_title, ch_content, verbose=verbose,
        )

        for nd in ch_nodes:
            counter += 1
            uid = f"{book.get('textbook_id','b')}_n{counter:04d}"
            nd["id"] = uid
            nd["chapter"] = ch_title
            nd["page"] = page_start
            nd["textbook"] = book.get("title", "")
            nd["textbook_id"] = book.get("textbook_id", "")
            name_to_id[nd.get("name", "").strip()] = uid
            nodes.append(nd)

        if progress_callback:
            progress_callback(
                (ci + 0.9) / total,
                f"[{ci+1}/{total}] 提取完成: {len(ch_nodes)} 个知识点"
            )

    if verbose:
        print(f"\n  📊 共提取 {len(nodes)} 个知识点，开始识别关系...")

    if progress_callback:
        progress_callback(0.98, "识别知识点间关系...")

    batch_size = 30
    for bi in range(0, len(nodes), batch_size):
        batch_nodes = nodes[bi:bi+batch_size]
        batch_relations = extract_relations(batch_nodes, verbose=verbose)
        for rel in batch_relations:
            s = name_to_id.get(rel.get("source", "").strip())
            t = name_to_id.get(rel.get("target", "").strip())
            if s and t and s != t:
                edges.append({
                    "source": s, "target": t,
                    "relation_type": rel.get("relation_type", "parallel"),
                    "description": rel.get("description", ""),
                })

    if progress_callback:
        progress_callback(1.0, f"✅ 完成: {len(nodes)} 节点, {len(edges)} 边")

    if verbose:
        print(f"  ✅ 图谱完成: {len(nodes)} 节点, {len(edges)} 关系\n")

    return {
        "textbook_id": book.get("textbook_id", ""),
        "title": book.get("title", ""),
        "nodes": nodes,
        "edges": edges,
        "stats": {"total_nodes": len(nodes), "total_edges": len(edges)},
    }


def merge_graphs(graphs: list) -> dict:
    """合并多个教材的图谱（保留来源信息）"""
    all_nodes, all_edges = [], []
    for g in graphs:
        all_nodes.extend(g.get("nodes", []))
        all_edges.extend(g.get("edges", []))
    return {
        "nodes": all_nodes, "edges": all_edges,
        "stats": {
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "textbook_count": len(graphs),
        }
    }


# ═══════════════════════════════════════════════
# ECharts 可视化格式（符合赛题要求）
# ═══════════════════════════════════════════════

TEXTBOOK_COLORS = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#48b8d0",
]

CAT_COLOR = {
    "核心概念": "#5470c6",
    "定理/定律": "#91cc75",
    "方法/技术": "#fac858",
    "现象/过程": "#ee6666",
    "结构/组成": "#73c0de",
}


def _get_textbook_color_map(nodes: list) -> dict:
    """为每个教材 ID 分配颜色"""
    textbook_ids = list(set(n.get("textbook_id", "") for n in nodes))
    color_map = {}
    for i, tid in enumerate(textbook_ids):
        color_map[tid] = TEXTBOOK_COLORS[i % len(TEXTBOOK_COLORS)]
    return color_map


def graph_to_echarts(graph: dict, max_nodes: int = 300) -> dict:
    """
    将图谱转为 ECharts 力导向图格式（赛题要求）
    
    - 节点颜色: 按教材来源区分
    - 节点大小: 按频次调整
    """
    all_nodes = graph.get("nodes", [])
    textbook_color_map = _get_textbook_color_map(all_nodes)

    name_freq = {}
    for n in all_nodes:
        name = n.get("name", "")
        name_freq[name] = name_freq.get(name, 0) + 1

    textbook_ids = list(textbook_color_map.keys())
    categories = []
    tid_to_cat_idx = {}
    for i, tid in enumerate(textbook_ids):
        book_name = ""
        for n in all_nodes:
            if n.get("textbook_id") == tid:
                book_name = n.get("textbook", "")
                break
        categories.append({"name": book_name or tid})
        tid_to_cat_idx[tid] = i

    limited_nodes = all_nodes[:max_nodes]
    node_ids = {n.get("id", "") for n in limited_nodes}

    echarts_nodes = []
    for n in limited_nodes:
        tid = n.get("textbook_id", "")
        freq = name_freq.get(n.get("name", ""), 1)
        symbol_size = min(60, max(20, 20 + freq * 8))

        echarts_nodes.append({
            "id": n.get("id", ""),
            "name": n.get("name", ""),
            "category": tid_to_cat_idx.get(tid, 0),
            "symbolSize": symbol_size,
            "itemStyle": {"color": textbook_color_map.get(tid, "#999")},
            "definition": n.get("definition", ""),
            "chapter": n.get("chapter", ""),
            "page": n.get("page", 1),
            "textbook": n.get("textbook", ""),
            "frequency": freq,
            "knowledge_category": n.get("category", "核心概念"),
        })

    echarts_links = []
    edge_count = 0
    for e in graph.get("edges", []):
        if e.get("source", "") in node_ids and e.get("target", "") in node_ids:
            if edge_count >= 500:
                break
            echarts_links.append({
                "source": e.get("source", ""),
                "target": e.get("target", ""),
                "label": {"show": True, "formatter": e.get("relation_type", ""), "fontSize": 10},
                "lineStyle": {"curveness": 0.2},
            })
            edge_count += 1

    return {
        "nodes": echarts_nodes,
        "links": echarts_links,
        "categories": categories,
        "stats": {
            "total_nodes": len(all_nodes),
            "displayed_nodes": len(echarts_nodes),
            "total_edges": len(graph.get("edges", [])),
            "displayed_edges": len(echarts_links),
        }
    }


def graph_to_echarts_integrated(
    graphs: list, decisions: list = None,
    mode: str = "merged", max_nodes: int = 300,
) -> dict:
    """整合后图谱 ECharts 格式（被合并节点灰色标记）"""
    merged = merge_graphs(graphs)

    removed_ids = set()
    if decisions and mode == "merged":
        for d in decisions:
            action = d.action if hasattr(d, 'action') else d.get('action', '')
            if str(action) in ('merge', 'remove'):
                affected = d.affected_nodes if hasattr(d, 'affected_nodes') \
                    else d.get('affected_nodes', [])
                for nid in affected:
                    removed_ids.add(nid)

    textbook_color_map = _get_textbook_color_map(merged.get("nodes", []))
    name_freq = {}
    for n in merged.get("nodes", []):
        name = n.get("name", "")
        name_freq[name] = name_freq.get(name, 0) + 1

    textbook_ids = list(textbook_color_map.keys())
    categories = []
    tid_to_idx = {}
    for i, tid in enumerate(textbook_ids):
        book_name = ""
        for n in merged.get("nodes", []):
            if n.get("textbook_id") == tid:
                book_name = n.get("textbook", "")
                break
        categories.append({"name": book_name or tid})
        tid_to_idx[tid] = i

    limited = merged.get("nodes", [])[:max_nodes]
    node_ids = {n.get("id", "") for n in limited}

    echarts_nodes = []
    for n in limited:
        tid = n.get("textbook_id", "")
        nid = n.get("id", "")
        freq = name_freq.get(n.get("name", ""), 1)
        is_removed = nid in removed_ids
        symbol_size = min(60, max(12 if is_removed else 20, 20 + freq * 8))
        color = textbook_color_map.get(tid, "#999")
        if is_removed:
            color = "#cccccc"

        echarts_nodes.append({
            "id": nid, "name": n.get("name", ""),
            "category": tid_to_idx.get(tid, 0),
            "symbolSize": symbol_size,
            "itemStyle": {"color": color},
            "definition": n.get("definition", ""),
            "chapter": n.get("chapter", ""),
            "page": n.get("page", 1),
            "textbook": n.get("textbook", ""),
            "frequency": freq,
            "knowledge_category": n.get("category", "核心概念"),
            "is_removed": is_removed,
        })

    echarts_links = []
    edge_set = set()
    for e in merged.get("edges", []):
        s, t = e.get("source", ""), e.get("target", "")
        if s in node_ids and t in node_ids:
            key = f"{s}->{t}"
            if key not in edge_set:
                edge_set.add(key)
                echarts_links.append({
                    "source": s, "target": t,
                    "label": {"show": True, "formatter": e.get("relation_type", "")},
                    "lineStyle": {"curveness": 0.2, "color": "#aaa"},
                })

    return {
        "nodes": echarts_nodes,
        "links": echarts_links,
        "categories": categories,
        "removed_count": len(removed_ids),
    }


if __name__ == "__main__":
    # 自测
    print("=" * 60)
    print("知识提取模块自测")
    print("=" * 60)
    assert _is_skip_chapter("前言 / 序言") == True
    assert _is_skip_chapter("第一章 绪论") == False
    assert _is_skip_chapter("主编简介") == True
    print("✅ _is_skip_chapter 正常")
    assert _clean_json('[{"a":1}]') == [{"a":1}]
    assert _clean_json('```json\n[{"a":1}]\n```') == [{"a":1}]
    assert _clean_json('前缀[{"a":1}]后缀') == [{"a":1}]
    print("✅ _clean_json 正常")
    print("\n所有自测通过!")


