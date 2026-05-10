"""
多轮对话管理模块 — AI 教师功能
支持教师通过自然语言对话：
  1. 询问整合决策的理由
  2. 对整合结果提出修改意见（保留/拆分/合并）
  3. 系统根据反馈调整整合方案

模型: Qwen3-0.6B (ModelScope) — 提示词需高度结构化
"""

import re, json
from dataclasses import dataclass, field
from llm_client import chat as llm_chat, call_llm


# ══════════════════════════════════════════════════════════
# System Prompt — 教师对话
# ══════════════════════════════════════════════════════════

TEACHER_SYSTEM_PROMPT = """# 身份
你是「学科知识整合智能体」的AI教师对话模块。你是一个专业的医学教育知识整合顾问。

# 背景
系统已将多本医学教材（如《生理学》《病理学》《传染病学》等）解析、构建知识图谱，并通过语义对齐算法完成了跨教材知识点整合。整合过程中产生了若干决策：合并重复知识点、保留唯一表述、删除冗余内容。

# 你的职责
1. 向教师解释每一项整合决策的理由（为什么合并、为什么删除）
2. 根据教师的反馈调整整合方案
3. 回答教师关于整合结果的任何问题

# 对话规则
- 始终用专业、清晰的中文回答
- 回答要有依据，引用具体的教材名和知识点名
- 如果教师提出修改意见，先确认理解，再说明你会如何调整
- 只讨论与教材整合、知识点、教学方法相关的话题
- 如果被问到无关话题（如天气、娱乐），礼貌拒绝并引导回正题
- 回答控制在 200 字以内，简洁有力

# 修改指令识别
当教师明确提出修改要求时，你需要识别以下意图：
- KEEP: 教师认为某个知识点不应被删除，应当保留
- SPLIT: 教师认为被合并的两个知识点应当分开
- MERGE: 教师认为两个未合并的知识点应当合并
- EXPLAIN: 教师询问某项决策的理由（默认，无需特殊标记）

对于 KEEP/SPLIT/MERGE 意图，在回复末尾附加一行 JSON：
```action
{"intent": "KEEP", "target": "知识点名称", "reason": "教师理由摘要"}
```"""


# ══════════════════════════════════════════════════════════
# Few-shot 示例（帮助小模型理解预期行为）
# ══════════════════════════════════════════════════════════

FEWSHOT_EXAMPLES = """## 对话示例

### 示例1：教师询问合并理由
教师：为什么把《生理学》的「炎症」和《病理学》的「炎症反应」合并了？
AI：这两处描述的是同一概念。《生理学》侧重炎症的生理机制（血管反应、白细胞渗出），《病理学》侧重炎症的病理变化和组织损伤。合并后保留了《病理学》的版本，因为其定义更完整，涵盖了病因、病理变化和临床表现三个维度。整合后节点名称为「炎症（病理学版）」。

### 示例2：教师要求保留被删除的知识点
教师：我觉得「免疫应答」不应该被删除，它和「免疫反应」不是同一个概念。
AI：理解您的意见。「免疫应答」强调的是机体对抗原的特异性识别和反应过程，而「免疫反应」范围更广。我会将「免疫应答」恢复为独立知识点，归属《医学微生物学》第六章。
```action
{"intent": "KEEP", "target": "免疫应答", "reason": "教师认为免疫应答与免疫反应概念不同，前者强调特异性识别过程"}
```

### 示例3：教师要求拆分
教师：把「抗原」和「免疫原」分开，它们不是同一个概念。
AI：您说得对。「抗原」指能诱导免疫应答的物质，侧重免疫原性和反应原性；「免疫原」特指能诱导免疫应答产生抗体/致敏淋巴细胞的物质，是抗原的子集。我会将合并的节点拆分为两个独立知识点。
```action
{"intent": "SPLIT", "target": "抗原/免疫原", "reason": "教师指出抗原和免疫原概念不同，前者是上位概念"}
```

### 示例4：无关话题
教师：今天天气怎么样？
AI：抱歉，我是学科知识整合顾问，只讨论教材整合和知识点相关的话题。请问您对整合结果有什么疑问或修改意见吗？"""


# ══════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════

@dataclass
class Message:
    role: str
    content: str


@dataclass
class DialogueSession:
    session_id: str
    messages: list = field(default_factory=list)
    integration_state: dict = field(default_factory=dict)
    pending_actions: list = field(default_factory=list)  # 待执行的修改动作

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def get_history(self) -> list:
        return [{"role": m.role, "content": m.content} for m in self.messages]


# ══════════════════════════════════════════════════════════
# 对话管理器
# ══════════════════════════════════════════════════════════

class DialogueManager:
    """AI 教师对话管理器"""

    MAX_HISTORY = 12  # 保留最近 12 轮对话（24 条消息）

    def __init__(self):
        self.sessions: dict[str, DialogueSession] = {}

    def get_or_create(self, sid: str) -> DialogueSession:
        if sid not in self.sessions:
            s = DialogueSession(session_id=sid)
            s.add("system", TEACHER_SYSTEM_PROMPT)
            s.add("user", FEWSHOT_EXAMPLES)
            s.add("assistant", "理解。我已准备好协助您审查整合决策。请随时提问。")
            self.sessions[sid] = s
        return self.sessions[sid]

    # ── 格式化整合状态供 LLM 理解 ────────────────

    @staticmethod
    def _format_decisions(decisions: list) -> str:
        """将整合决策列表格式化为 LLM 可读文本"""
        if not decisions:
            return "（暂无整合决策数据）"

        lines = ["## 当前整合决策一览\n"]
        for i, d in enumerate(decisions, 1):
            if isinstance(d, dict):
                action = d.get("action", "?")
                reason = d.get("reason", "")
                affected = d.get("affected_nodes", [])
                confidence = d.get("confidence", 0)
            else:
                action = getattr(d, "action", "?")
                reason = getattr(d, "reason", "")
                affected = getattr(d, "affected_nodes", [])
                confidence = getattr(d, "confidence", 0)

            action_cn = {"merge": "合并", "keep": "保留", "remove": "删除"}.get(
                str(action), str(action)
            )
            nodes_str = "、".join(str(n)[:60] for n in affected[:3])
            lines.append(
                f"{i}. [{action_cn}] {reason}"
                + (f" | 涉及: {nodes_str}" if nodes_str else "")
                + f" | 置信度: {confidence:.0%}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_textbook_summary(books: dict) -> str:
        """将教材信息格式化为摘要"""
        if not books:
            return "（暂无教材信息）"
        lines = ["## 已加载教材\n"]
        for bid, b in books.items():
            title = getattr(b, "title", str(b)[:30])
            pages = getattr(b, "total_pages", "?")
            chs = len(getattr(b, "chapters", []))
            chars = getattr(b, "total_chars", 0)
            lines.append(f"- 《{title}》| {pages}页 | {chs}章 | {chars:,}字")
        return "\n".join(lines)

    # ── 核心对话方法 ─────────────────────────────

    def chat(self, sid: str, user_msg: str, decisions: list = None,
             books: dict = None) -> str:
        """处理一轮教师对话

        Args:
            sid: 会话ID
            user_msg: 教师输入
            decisions: 当前整合决策列表
            books: 已加载教材字典 {id: TextbookInfo}

        Returns:
            AI 回复文本（可能包含 ```action JSON 块）
        """
        session = self.get_or_create(sid)

        # 更新整合状态
        if decisions is not None:
            session.integration_state = {
                "decisions": [
                    d.to_dict() if hasattr(d, "to_dict") else d
                    for d in decisions
                ]
            }
        if books is not None:
            session.integration_state["books"] = books

        session.add("user", user_msg)

        # ── 构建上下文 ──
        context_parts = []

        # 教材摘要
        state_books = session.integration_state.get("books")
        if state_books or books:
            context_parts.append(
                self._format_textbook_summary(state_books or books)
            )

        # 整合决策摘要
        decisions_data = (
            session.integration_state.get("decisions") or decisions or []
        )
        if decisions_data:
            context_parts.append(self._format_decisions(decisions_data))

        context = "\n\n".join(context_parts) if context_parts else ""

        # ── 构建消息列表 ──
        # system prompt 合并 few-shot 示例
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT + "\n\n" + FEWSHOT_EXAMPLES}
        ]

        # 对话历史（跳过内部的 system + 前两条 few-shot 消息）
        history = session.messages[3:]  # skip system, fewshot_user, fewshot_asst
        if len(history) > self.MAX_HISTORY:
            history = history[-self.MAX_HISTORY:]
        for m in history:
            messages.append({"role": m.role, "content": m.content})

        # 当前轮：在最后一条 user 消息前注入上下文
        # (最后一条 user 消息就是刚添加的 user_msg)
        if context:
            messages[-1]["content"] = (
                f"{context}\n\n---\n教师：{user_msg}"
            )

        # ── 调用 LLM ──
        reply = llm_chat(messages, temperature=0.5)

        # ── 解析 action JSON ──
        action = self._parse_action(reply)
        if action:
            session.pending_actions.append(action)

        session.add("assistant", reply)
        return reply

    # ── Action 解析 ──────────────────────────────

    @staticmethod
    def _parse_action(reply: str) -> dict | None:
        """从回复中提取 ```action JSON 块"""
        m = re.search(r'```action\s*\n(.*?)\n```', reply, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    # ── 获取待执行的修改 ─────────────────────────

    def get_pending_actions(self, sid: str) -> list:
        """获取某个会话中待执行的修改动作列表"""
        session = self.sessions.get(sid)
        if not session:
            return []
        return session.pending_actions

    def clear_pending_actions(self, sid: str):
        """清空待执行动作"""
        session = self.sessions.get(sid)
        if session:
            session.pending_actions.clear()

    # ── 会话管理 ─────────────────────────────────

    def reset_session(self, sid: str):
        """重置会话（保留整合状态，清空对话历史）"""
        if sid in self.sessions:
            decisions = self.sessions[sid].integration_state.get("decisions", [])
            books = self.sessions[sid].integration_state.get("books", {})
            del self.sessions[sid]
            s = self.get_or_create(sid)
            s.integration_state = {"decisions": decisions, "books": books}

    def get_conversation(self, sid: str) -> list:
        """获取对话历史（不含 system prompt 和 few-shot）"""
        session = self.sessions.get(sid)
        if not session:
            return []
        # 跳过 system + few-shot user + few-shot assistant
        return [{"role": m.role, "content": m.content}
                for m in session.messages[3:]]
