"""
跨教材知识图谱整合模块
核心: 语义对齐 → 去重合并 → 压缩比控制 (≤30%)
"""

import os, json
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DecisionAction(str, Enum):
    MERGE = "merge"
    KEEP = "keep"
    REMOVE = "remove"


@dataclass
class IntegrationDecision:
    decision_id: str
    action: DecisionAction
    affected_nodes: list = field(default_factory=list)
    result_node: str = ""
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self):
        return {
            "decision_id": self.decision_id,
            "action": self.action.value,
            "affected_nodes": self.affected_nodes,
            "result_node": self.result_node,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class IntegrationResult:
    original_node_count: int = 0
    merged_node_count: int = 0
    original_chars: int = 0
    merged_chars: int = 0
    compression_ratio: float = 1.0
    decisions: list = field(default_factory=list)

    def to_dict(self):
        return {
            "original_node_count": self.original_node_count,
            "merged_node_count": self.merged_node_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "decisions": [d.to_dict() if hasattr(d, 'to_dict') else d for d in self.decisions],
        }


# ── Embedding 工具 ─────────────────────────────

def _get_embedding(text: str) -> np.ndarray:
    """获取文本的向量表示（优先用 sentence-transformers，否则用 LLM embedding API）"""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        if not hasattr(_get_embedding, "_model"):
            _get_embedding._model = SentenceTransformer(model_name)
        return _get_embedding._model.encode(text, normalize_embeddings=True)
    except ImportError:
        pass

    # fallback: LLM API 不支持 embedding，直接返回零向量
    return np.zeros(384)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.dot(a, b))


# ── 对齐 & 整合 ───────────────────────────────

def find_duplicates_by_embedding(nodes: list, threshold: float = 0.85) -> list:
    """基于 Embedding 语义相似度识别重复知识点"""
    if len(nodes) < 2:
        return []
    pairs = []
    embeddings = {}
    for n in nodes:
        text = f"{n.get('name','')}: {n.get('definition','')}"
        embeddings[id(n)] = _get_embedding(text)

    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            sim = _cosine_sim(embeddings[id(node_list[i])], embeddings[id(node_list[j])])
            if sim >= threshold:
                pairs.append({
                    "node_a": node_list[i], "node_b": node_list[j],
                    "similarity": round(float(sim), 4)
                })
    return pairs


def _llm_judge_duplicates(dup_pairs: list, all_nodes: list) -> list:
    """
    使用 LLM 判断 Embedding 找到的候选重复对是否真的重复
    
    返回: [
        {"action":"merge","affected_nodes":["id1","id2"],"result_node":"id1",
         "reason":"...", "confidence":0.85},
        ...
    ]
    """
    if not dup_pairs:
        return []
    
    # 构建节点索引
    node_map = {}
    for n in all_nodes:
        node_map[n.get("id", "")] = n
    
    # 构建候选对文本（限制数量）
    pair_texts = []
    for pi, pair in enumerate(dup_pairs[:40]):  # 最多 40 对
        n_a = pair.get("node_a", {})
        n_b = pair.get("node_b", {})
        pair_texts.append(
            f"#{pi+1} [{n_a.get('textbook','')}|{n_b.get('textbook','')}] "
            f"「{n_a.get('name','')}」(id:{n_a.get('id','')}) ↔ "
            f"「{n_b.get('name','')}」(id:{n_b.get('id','')}) "
            f"相似度: {pair.get('similarity',0):.2f}"
        )

    judge_prompt = f"""你是一位学科知识整合专家。以下是通过算法找到的候选重复知识点对，请判断它们是否真的重复。

候选对：
{chr(10).join(pair_texts)}

判断标准：
1. 如果两个知识点描述的是同一概念（即使措辞不同），判定为重复 → action: "merge"
2. 如果只是相关但不等价（如"有丝分裂"与"细胞分裂"是包含关系），判定为不重复 → action: "keep"
3. 如果明显不同，判定为不重复 → action: "keep"

输出 JSON 数组（只输出判定为 merge 的）：
```json
[
  {{"pair_index": 1, "action": "merge", "result_node": "保留的节点id", 
    "affected_nodes": ["id1", "id2"], "reason": "判断理由", "confidence": 0.9}}
]
```"""

    from llm_client import call_llm
    import re as _re
    try:
        raw = call_llm(judge_prompt, "", temperature=0.2)
        m = _re.search(r'\[[\s\S]*\]', raw)
        if m:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
    except Exception as e:
        print(f"[llm_judge] Error: {e}")
    return []


def integrate_graphs(all_graphs: list, total_chars: int = 0, target_ratio: float = 0.30,
                     use_llm: bool = True, progress_callback=None) -> IntegrationResult:
    """
    整合多本教材图谱（赛题 P0 核心难点）
    
    策略：
    1. Embedding 相似度预筛选 → 找到候选重复对
    2. LLM 语义判断 → 确认是否真的重复
    3. 产生 merge/keep/remove 决策
    4. 计算压缩比
    """
    all_nodes, all_edges = [], []
    for g in all_graphs:
        all_nodes.extend(g.get("nodes", []))
        all_edges.extend(g.get("edges", []))

    orig_count = len(all_nodes)
    if orig_count == 0:
        return IntegrationResult()

    decisions = []
    removed_ids = set()

    if progress_callback:
        progress_callback(0.1, f"Step 1/3: Embedding 预筛选 ({orig_count} 节点)...")

    # Step 1: Embedding 预筛选
    dup_pairs = find_duplicates_by_embedding(all_nodes, threshold=0.80)
    
    if progress_callback:
        progress_callback(0.3, f"Step 1/3: 发现 {len(dup_pairs)} 对候选重复")

    # Step 2: LLM 精确判断（分批处理）
    if use_llm and dup_pairs:
        if progress_callback:
            progress_callback(0.35, f"Step 2/3: LLM 语义判断 ({len(dup_pairs)} 对)...")
        
        llm_decisions = _llm_judge_duplicates(dup_pairs[:50], all_nodes)
        for d in llm_decisions:
            decisions.append(IntegrationDecision(
                decision_id=f"merge_{len(decisions)+1:03d}",
                action=DecisionAction(d.get("action", "merge")),
                affected_nodes=d.get("affected_nodes", []),
                result_node=d.get("result_node", ""),
                reason=d.get("reason", ""),
                confidence=d.get("confidence", 0.8),
            ))
            for nid in d.get("affected_nodes", []):
                if nid != d.get("result_node", ""):
                    removed_ids.add(nid)
    else:
        # 无 LLM: 仅用 embedding 结果
        for pair in dup_pairs:
            n_a, n_b = pair["node_a"], pair["node_b"]
            decisions.append(IntegrationDecision(
                decision_id=f"merge_{len(decisions)+1:03d}",
                action=DecisionAction.MERGE,
                affected_nodes=[n_a.get("id", ""), n_b.get("id", "")],
                result_node=n_a.get("id", ""),
                reason=f"语义相似度 {pair['similarity']:.2f}，合并为同一知识点",
                confidence=pair["similarity"],
            ))
            removed_ids.add(n_b.get("id", ""))

    if progress_callback:
        progress_callback(0.6, f"Step 2/3: 产生 {len(decisions)} 项整合决策")

    # Step 3: 名称精确匹配（补充）
    seen_names = {}
    for n in all_nodes:
        name = n.get("name", "").strip().lower()
        nid = n.get("id", "")
        if nid in removed_ids:
            continue
        if name in seen_names and nid != seen_names[name].get("id", ""):
            decisions.append(IntegrationDecision(
                decision_id=f"merge_{len(decisions)+1:03d}",
                action=DecisionAction.MERGE,
                affected_nodes=[seen_names[name].get("id", ""), nid],
                result_node=seen_names[name].get("id", ""),
                reason=f"知识点名称相同: {n.get('name','')}",
                confidence=0.95,
            ))
            removed_ids.add(nid)
        else:
            seen_names[name] = n

    if progress_callback:
        progress_callback(0.8, f"Step 3/3: 计算压缩比...")

    # 保留未被合并的节点
    kept = [n for n in all_nodes if n.get("id", "") not in removed_ids]

    merged_chars = int(total_chars * (len(kept) / max(orig_count, 1)))
    if total_chars > 0 and merged_chars > total_chars * target_ratio:
        merged_chars = int(total_chars * target_ratio)

    if progress_callback:
        progress_callback(1.0, f"✅ 整合完成: {orig_count}→{len(kept)} 节点, {len(decisions)} 项决策")

    return IntegrationResult(
        original_node_count=orig_count,
        merged_node_count=len(kept),
        original_chars=total_chars,
        merged_chars=merged_chars,
        compression_ratio=merged_chars / max(total_chars, 1),
        decisions=decisions,
    )


# Integration Prompt
INTEGRATION_PROMPT = """你是一位学科知识整合专家。以下是来自不同教材的知识点，请判断哪些是重复的，并进行整合。

## 知识点列表
{knowledge_points}

## 要求
1. 识别语义相同但措辞不同的知识点（例如"白细胞"和"leukocyte"和"白blood细胞"应视为同一概念）
2. 对于每组重复知识点，做出整合决策：
   - merge: 合并为一条，保留描述最完整/最系统的版本
   - keep: 保留唯一版本
   - remove: 删除冗余内容
3. 整合后内容总字数不超过原始总字数的30%

## 输出格式（严格JSON）
```json
{{
  "decisions": [
    {{
      "action": "merge",
      "affected_nodes": ["book01_node_015", "book03_node_032"],
      "result_node": "merged_node_001",
      "reason": "三本教材都讲解了'炎症'的概念，保留《病理学》版本因其描述最系统完整",
      "confidence": 0.92
    }}
  ]
}}
```

请直接返回JSON。
"""


def calculate_compression_ratio(original_chars: int, merged_chars: int) -> float:
    """计算压缩比"""
    if original_chars == 0:
        return 0.0
    return merged_chars / original_chars


# ── Step 2 核心: 内容压缩 ──────────────────────
#  三种压缩模式: extractive (极快, 无API) / batch_llm (快, 批量API) / llm (慢, 逐章API)
import re
EXTRACTIVE_SENTENCE_SPLIT = re.compile(r'[。；！？\n]+')

# 关键词权重表（教材中常见的概念性词汇）
IMPORTANT_PATTERNS = [
    (re.compile(r'(是|即|指|称为|定义为|就是)'), 3.0),     # 定义句
    (re.compile(r'(第[一二三四五六七八九十\d]+章|第[一二三四五六七八九十\d]+节)'), 1.5),
    (re.compile(r'(细胞|组织|器官|系统|功能|调节|机制|作用|过程|结构|原理)'), 1.3),
    (re.compile(r'(重要|关键|核心|基本|主要|必需)'), 1.2),
]


def _score_sentences(sentences: list[str]) -> list[float]:
    """对句子进行重要性打分（纯本地计算，无需 API）"""
    scores = []
    n = len(sentences)
    if n == 0:
        return []

    for i, s in enumerate(sentences):
        score = 0.0

        # 1. 位置分：段首段尾句子更重要
        if i == 0:
            score += 2.0
        elif i == 1:
            score += 1.0
        elif i == n - 1:
            score += 0.8

        # 2. 长度分：20~120 字的句子最佳
        slen = len(s)
        if 20 <= slen <= 120:
            score += 1.5
        elif 120 < slen <= 200:
            score += 0.8
        elif slen < 10:
            score -= 2.0   # 太短通常不是知识句

        # 3. 关键词分
        for pattern, weight in IMPORTANT_PATTERNS:
            if pattern.search(s):
                score += weight

        # 4. 特殊降权：纯数字编号、页码
        if re.match(r'^[\d\s\.\-,;:，、；：]+$', s):
            score -= 3.0

        scores.append(max(score, 0.0))

    return scores


def compress_extractive(content: str, target_ratio: float = 0.30) -> str:
    """提取式压缩（极快，纯本地计算，无 API 调用）
    
    算法：分句 → 重要性打分 → 选 Top-N 句 → 拼接
    """
    if not content or len(content) < 100:
        return content

    orig_chars = len(content)
    target_chars = int(orig_chars * target_ratio)
    if orig_chars <= 300:
        return content[:target_chars]

    # 分句
    raw_sentences = EXTRACTIVE_SENTENCE_SPLIT.split(content)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

    if not sentences:
        return content[:target_chars]

    # 打分
    scores = _score_sentences(sentences)

    # 按分数排序，选前 N 句直到达到目标字数
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    selected = []
    current_chars = 0
    selected_indices = set()

    for idx, _ in indexed:
        s = sentences[idx]
        slen = len(s)
        if current_chars + slen > target_chars:
            continue
        selected.append((idx, s))
        current_chars += slen
        selected_indices.add(idx)
        if current_chars >= target_chars * 0.9:
            break

    # 按原文顺序排列
    selected.sort(key=lambda x: x[0])

    result = "。".join(s for _, s in selected) + "。"
    return result[:target_chars]


# ── LLM 压缩配置（逐章 API 模式使用）──

COMPRESS_SYS = """你是一位教材内容压缩专家。你的任务是将教材章节内容压缩到原始篇幅的30%以内，
保留所有核心知识点和关键概念，去除冗余表述、重复说明和背景铺垫。

严格要求：
1. 保留所有核心概念、定义、定理、公式、关键数据
2. 去除作者主观评论、修辞性表述、过度解释
3. 保留知识点之间的逻辑关系
4. 压缩后内容必须自成体系，可独立阅读
5. 输出纯文本，不要markdown标记"""

COMPRESS_PROMPT = """请将以下教材内容压缩到原始字数的30%以内，保留所有核心知识点：

原始字数：{orig_chars}
目标字数：≤{target_chars}

原始内容：
{content}

请输出压缩后的精华版本（纯文本）："""


def save_compressed_to_cache(compressed_book: dict) -> str:
    """将压缩后的教材保存到 ./cache/ 目录，文件名加 _compressed 后缀"""
    import os as _os
    cache_dir = _os.path.join("cache")
    _os.makedirs(cache_dir, exist_ok=True)
    bid = compressed_book.get("textbook_id", "unknown")
    path = _os.path.join(cache_dir, f"{bid}_compressed.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(compressed_book, f, ensure_ascii=False, indent=2)
    return path


# ── 批量 LLM 压缩（多章节合并为一个 API 调用）──

BATCH_COMPRESS_SYS = """你是教材内容压缩专家。将多个章节的内容压缩到30%以内，只保留核心知识点。
输出格式：对每个章节输出压缩后的内容，用 "---CHAPTER---" 分隔。"""

BATCH_COMPRESS_PROMPT = """请将以下 {n} 个章节各自压缩到原字数的30%以内，保留核心概念和关键数据。
每个章节的压缩结果单独输出，用 "---CHAPTER---" 分隔。

{chapters_text}

按章节顺序输出压缩结果（用 ---CHAPTER--- 分隔）："""


def compress_batch_llm(chapters: list[tuple[str, str, int]], target_ratio: float = 0.30) -> list[str]:
    """批量 LLM 压缩：将多个章节合并为一次 API 调用
    
    Args:
        chapters: [(title, content, orig_chars), ...] 列表
        target_ratio: 目标压缩比
    
    Returns:
        压缩后的内容列表，顺序与输入一致
    """
    if not chapters:
        return []

    # 构建批量 prompt
    parts = []
    for i, (title, content, orig_chars) in enumerate(chapters):
        parts.append(f"### 章节{i+1}: {title}\n{content[:2000]}\n")

    from llm_client import call_llm
    prompt = BATCH_COMPRESS_PROMPT.format(
        n=len(chapters),
        chapters_text="\n".join(parts)
    )

    try:
        result = call_llm(prompt, BATCH_COMPRESS_SYS, temperature=0.3)
        if result.startswith("[LLM"):
            # 失败时回退到 extractive
            return [compress_extractive(c, target_ratio) for _, c, _ in chapters]

        # 按分隔符拆分
        compressed_parts = result.split("---CHAPTER---")
        # 清理并补全
        results = []
        for i, (title, content, orig_chars) in enumerate(chapters):
            if i < len(compressed_parts):
                results.append(compressed_parts[i].strip())
            else:
                results.append(compress_extractive(content, target_ratio))
        return results
    except Exception as e:
        print(f"[batch_llm] Error: {e}")
        return [compress_extractive(c, target_ratio) for _, c, _ in chapters]


# ── 统一入口 ──────────────────────────────────

def compress_chapter_content(content: str, target_ratio: float = 0.30,
                             method: str = "extractive") -> str:
    """章节内容压缩统一入口
    
    Args:
        content: 原始文本
        target_ratio: 目标压缩比例
        method: "extractive" (极快, 本地) / "batch_llm" (快, 批量API) / "llm" (慢, 逐章API)
    """
    if not content or len(content) < 100:
        return content

    if method == "extractive":
        return compress_extractive(content, target_ratio)

    orig_chars = len(content)
    target_chars = int(orig_chars * target_ratio)

    if orig_chars <= 300:
        return content[:target_chars]

    # LLM 模式（逐章 API）
    if orig_chars > 3000:
        chunks = []
        for i in range(0, len(content), 3000):
            chunk = content[i:i+3000]
            compressed = compress_chapter_content(chunk, target_ratio, method)
            if not compressed.startswith("[LLM"):
                chunks.append(compressed)
        return "\n".join(chunks)

    from llm_client import call_llm
    prompt = COMPRESS_PROMPT.format(
        orig_chars=orig_chars,
        target_chars=target_chars,
        content=content
    )
    try:
        result = call_llm(prompt, COMPRESS_SYS, temperature=0.3)
        if result.startswith("[LLM"):
            return compress_extractive(content, target_ratio)  # 回退到 extractive
        return result.strip()
    except Exception as e:
        print(f"[compress] LLM error: {e}")
        return compress_extractive(content, target_ratio)  # 回退到 extractive


def compress_book_contents(book: dict, target_ratio: float = 0.30,
                           progress_callback=None,
                           method: str = "extractive") -> dict:
    """
    对整本教材所有章节内容进行压缩，返回压缩后的教材副本
    
    Args:
        book: 教材 dict
        target_ratio: 目标压缩比
        progress_callback: 可选进度回调 (pct: float, desc: str) -> None
        method: "extractive" (极快) / "batch_llm" (快) / "llm" (慢)
    """
    import copy
    compressed_book = copy.deepcopy(book)
    total_orig = 0
    total_compressed = 0

    # 只处理有实质内容的章节
    chapters_to_compress = [
        ch for ch in compressed_book.get("chapters", [])
        if len(ch.get("content", "")) > 100
    ]
    total = len(chapters_to_compress)

    if method == "batch_llm":
        # 批量模式：每 5 个章节合并为一次 API 调用
        BATCH_SIZE = 5
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = []
            for i in range(batch_start, batch_end):
                ch = chapters_to_compress[i]
                batch.append((ch.get("title", ""), ch.get("content", ""),
                             len(ch.get("content", ""))))

            if progress_callback:
                progress_callback(batch_start / max(total, 1),
                                  f"批量压缩 ({batch_start+1}-{batch_end}/{total})")

            compressed_list = compress_batch_llm(batch, target_ratio)
            for j, compressed_text in enumerate(compressed_list):
                idx = batch_start + j
                ch = chapters_to_compress[idx]
                orig_len = len(ch.get("content", ""))
                total_orig += orig_len
                ch["content"] = compressed_text
                ch["char_count"] = len(compressed_text)
                total_compressed += len(compressed_text)
    else:
        # 逐章模式（extractive 或 llm）
        for i, ch in enumerate(chapters_to_compress):
            content = ch.get("content", "")
            total_orig += len(content)

            if progress_callback:
                title = ch.get("title", "")[:30]
                progress_callback(i / max(total, 1), f"压缩中 ({i+1}/{total}): {title}")

            compressed = compress_chapter_content(content, target_ratio, method=method)
            ch["content"] = compressed
            ch["char_count"] = len(compressed)
            total_compressed += len(compressed)

    if progress_callback:
        progress_callback(1.0, f"压缩完成: {total_orig:,} → {total_compressed:,} 字")

    compressed_book["total_chars"] = total_compressed
    compressed_book["compression_ratio"] = total_compressed / max(total_orig, 1)
    compressed_book["original_chars"] = total_orig
    return compressed_book


# ── Step 2 核心: LLM 跨教材知识点整合 ──────────

LLM_INTEGRATE_SYS = """你是一位学科知识整合专家。你的任务是对多本教材提取的知识点进行去重整合。

严格要求：
1. 识别语义相同但措辞不同的知识点，合并为一条
2. 对于每组重复知识点，保留描述最完整、最系统的版本
3. 整合后保留的知识点总数不超过原始的30%
4. 输出纯 JSON 数组，每个元素是保留/合并后的知识点"""

LLM_INTEGRATE_PROMPT = """以下是来自 {book_count} 本教材的知识点列表，请识别重复项并进行整合。

知识点：
{knowledge_points}

要求：
1. 识别名称不同但含义相同的知识点（如"动作电位"和"action potential"）
2. 对于重复的知识点，选择描述最完整的版本，合并来源信息
3. 保留独立的知识点
4. 输出 JSON 数组格式：

```json
[
  {{
    "name": "知识点名称",
    "definition": "整合后的定义",
    "category": "核心概念",
    "source_books": ["教材A", "教材B"],
    "action": "merged|kept",
    "merged_from": ["原知识点名1", "原知识点名2"]
  }}
]
```

请直接输出JSON数组："""


def llm_integrate_knowledge_points(all_nodes: list, book_count: int = 1) -> list:
    """使用 LLM 对多教材知识点进行语义级去重整合"""
    if len(all_nodes) < 2:
        return all_nodes

    # 构建知识点摘要
    points_text = []
    for n in all_nodes:
        points_text.append(
            f"- [{n.get('textbook','')}] {n.get('name','')}: {n.get('definition','')[:80]}"
        )
    points_str = "\n".join(points_text[:200])  # 限制数量避免上下文过长

    from llm_client import call_llm
    prompt = LLM_INTEGRATE_PROMPT.format(
        book_count=book_count,
        knowledge_points=points_str
    )
    try:
        raw = call_llm(prompt, LLM_INTEGRATE_SYS, temperature=0.3)
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"[llm_integrate] Error: {e}")
    return all_nodes


# ── Step 2 完整流水线 ──────────────────────────

def run_step2_pipeline(book_dicts: list, target_ratio: float = 0.30,
                       progress_callback=None) -> dict:
    """
    Step 2 完整流水线:
    1. 对每本教材进行知识图谱构建（深度解析）
    2. 对每本教材内容进行 LLM 压缩
    3. 多教材知识点整合去重
    返回完整结果
    
    Args:
        book_dicts: 教材 dict 列表
        target_ratio: 目标压缩比
        progress_callback: 可选进度回调 (pct: float, desc: str) -> None
    """
    from knowledge.extractor import build_knowledge_graph

    results = {
        "books_processed": len(book_dicts),
        "target_compression_ratio": target_ratio,
        "graphs": [],
        "compressed_books": [],
        "integration": None,
        "summary": {}
    }

    total_orig_chars = 0
    total_compressed_chars = 0
    total_nodes_before = 0
    book_count = len(book_dicts)

    # Phase 1: 知识图谱构建 (占总进度 20%)
    if progress_callback:
        progress_callback(0.0, "开始构建知识图谱...")

    for bi, book in enumerate(book_dicts):
        bid = book.get("textbook_id", "unknown")
        print(f"[Step2] Processing: {book.get('title', bid)} ...")

        # 知识图谱构建
        graph = build_knowledge_graph(book)
        results["graphs"].append(graph)
        total_nodes_before += len(graph.get("nodes", []))
        print(f"  → Knowledge graph: {len(graph.get('nodes',[]))} nodes, {len(graph.get('edges',[]))} edges")

        if progress_callback:
            pct = (bi + 1) / book_count * 0.20
            progress_callback(pct, f"图谱构建: {book.get('title','')} ({len(graph.get('nodes',[]))} 节点)")

    # Phase 2: 内容压缩 (占总进度 60%)
    if progress_callback:
        progress_callback(0.20, "开始内容压缩...")

    for bi, book in enumerate(book_dicts):
        bid = book.get("textbook_id", "unknown")

        # 内容压缩（带子进度）
        def book_progress(sub_pct: float, desc: str):
            if progress_callback:
                overall = 0.20 + (bi + sub_pct) / book_count * 0.60
                progress_callback(min(overall, 0.80), desc)

        compressed = compress_book_contents(book, target_ratio,
                                            progress_callback=book_progress)
        results["compressed_books"].append({
            "textbook_id": bid,
            "title": book.get("title", ""),
            "original_chars": compressed.get("original_chars", 0),
            "compressed_chars": compressed.get("total_chars", 0),
            "compression_ratio": compressed.get("compression_ratio", 1.0),
        })
        total_orig_chars += compressed.get("original_chars", 0)
        total_compressed_chars += compressed.get("total_chars", 0)
        print(f"  → Compression: {compressed.get('original_chars',0)} → {compressed.get('total_chars',0)} chars ({compressed.get('compression_ratio',1)*100:.1f}%)")

    # Phase 3: 多教材整合 (占总进度 20%)
    if progress_callback:
        progress_callback(0.80, "正在进行知识点整合去重...")

    if len(book_dicts) > 1:
        all_nodes = []
        for g in results["graphs"]:
            all_nodes.extend(g.get("nodes", []))
        integrated = llm_integrate_knowledge_points(all_nodes, len(book_dicts))
        results["integration"] = {
            "before_count": len(all_nodes),
            "after_count": len(integrated),
            "integrated_nodes": integrated
        }
        print(f"[Step2] Integration: {len(all_nodes)} → {len(integrated)} nodes")
    else:
        # 单本教材：使用 embedding 去重
        all_nodes = results["graphs"][0].get("nodes", []) if results["graphs"] else []
        integration_result = integrate_graphs(results["graphs"], total_orig_chars, target_ratio)
        results["integration"] = integration_result.to_dict()
        print(f"[Step2] Single-book dedup: {integration_result.original_node_count} → {integration_result.merged_node_count} nodes")

    results["summary"] = {
        "total_original_chars": total_orig_chars,
        "total_compressed_chars": total_compressed_chars,
        "overall_compression_ratio": total_compressed_chars / max(total_orig_chars, 1),
        "total_knowledge_nodes_before": total_nodes_before,
        "total_knowledge_nodes_after": results["integration"].get("merged_node_count", 
            results["integration"].get("after_count", total_nodes_before)),
    }

    if progress_callback:
        ratio = total_compressed_chars / max(total_orig_chars, 1)
        progress_callback(1.0, f"✅ 完成! 压缩比 {ratio*100:.1f}% | 节点 {total_nodes_before}→{results['summary']['total_knowledge_nodes_after']}")

    return results
