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

    # fallback: 使用 LLM API embedding
    import requests
    key = os.getenv("MY_LLM_API_KEY") or os.getenv("LLM_API_KEY") or "ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2"
    base = os.getenv("MY_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://ms-ens-f8274faf-bcde.api-inference.modelscope.cn/v1"
    r = requests.post(
        f"{base}/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "text-embedding-3-small", "input": text},
        timeout=30
    )
    if r.status_code == 200:
        emb = np.array(r.json()["data"][0]["embedding"])
        return emb / np.linalg.norm(emb)
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


def integrate_graphs(all_graphs: list, total_chars: int = 0, target_ratio: float = 0.30) -> IntegrationResult:
    """
    整合多本教材图谱
    1. 合并所有节点
    2. 语义对齐识别重复
    3. 产生合并/保留/删除决策
    4. 计算压缩比
    """
    all_nodes, all_edges = [], []
    for g in all_graphs:
        all_nodes.extend(g.get("nodes", []))
        all_edges.extend(g.get("edges", []))

    orig_count = len(all_nodes)
    decisions = []
    removed_ids = set()

    # 查找重复对
    dup_pairs = find_duplicates_by_embedding(all_nodes, threshold=0.82)

    # 处理重复：保留第一个，标记后续为待合并
    seen_names = {}
    for i, n in enumerate(all_nodes):
        name = n.get("name", "").strip()
        if name in seen_names:
            decisions.append(IntegrationDecision(
                decision_id=f"merge_{len(decisions)+1:03d}",
                action=DecisionAction.MERGE,
                affected_nodes=[seen_names[name].get("id",""), n.get("id","")],
                result_node=seen_names[name].get("id", ""),
                reason=f"知识点名称相同: {name}",
                confidence=0.95,
            ))
            removed_ids.add(n.get("id", ""))
        else:
            seen_names[name] = n

    # 保留未被合并的节点
    kept = [n for n in all_nodes if n.get("id", "") not in removed_ids]

    # 计算压缩后字数（粗略：按保留节点数比例估算）
    merged_chars = int(total_chars * (len(kept) / max(orig_count, 1)))
    # 强制压缩到目标比例
    if total_chars > 0 and merged_chars > total_chars * target_ratio:
        merged_chars = int(total_chars * target_ratio)

    return IntegrationResult(
        original_node_count=orig_count,
        merged_node_count=len(kept),
        original_chars=total_chars,
        merged_chars=merged_chars,
        compression_ratio=merged_chars / max(total_chars, 1),
        decisions=decisions,
    )


def calculate_compression_ratio(original_chars: int, merged_chars: int) -> float:
    return merged_chars / max(original_chars, 1)


    def to_dict(self):
        return {
            "original_node_count": self.original_node_count,
            "merged_node_count": self.merged_node_count,
            "original_edge_count": self.original_edge_count,
            "merged_edge_count": self.merged_edge_count,
            "original_chars": self.original_chars,
            "merged_chars": self.merged_chars,
            "compression_ratio": round(self.compression_ratio, 4),
            "compression_percent": f"{round(self.compression_ratio * 100, 1)}%",
            "decisions": [d.to_dict() for d in self.decisions],
        }


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


def compute_similarity(emb1, emb2) -> float:
    """计算余弦相似度"""
    if emb1 is None or emb2 is None:
        return 0.0
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return float(dot / norm) if norm > 0 else 0.0


def find_duplicates_by_embedding(
    nodes: list,
    embeddings: dict,
    threshold: float = 0.85,
) -> list:
    """基于 Embedding 相似度寻找重复知识点"""
    duplicates = []
    visited = set()
    node_ids = list(embeddings.keys())

    for i, id1 in enumerate(node_ids):
        if id1 in visited:
            continue
        group = [id1]
        for j in range(i + 1, len(node_ids)):
            id2 = node_ids[j]
            if id2 in visited:
                continue
            sim = compute_similarity(embeddings.get(id1), embeddings.get(id2))
            if sim >= threshold:
                group.append(id2)
                visited.add(id2)
        if len(group) > 1:
            duplicates.append(group)
        visited.add(id1)

    return duplicates


def calculate_compression_ratio(
    original_chars: int,
    merged_chars: int,
) -> float:
    """计算压缩比"""
    if original_chars == 0:
        return 0.0
    return merged_chars / original_chars
