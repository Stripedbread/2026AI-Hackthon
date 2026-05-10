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
from .retriever import KeywordRetriever, extract_keywords


# ══════════════════════════════════════════════════════════
# System Prompt — 教师对话
# ══════════════════════════════════════════════════════════

TEACHER_SYSTEM_PROMPT = """# 身份
你是「学科知识整合智能体」的AI教师对话模块。你是一个专业的医学教育知识整合顾问。

# 背景
系统已将多本医学教材（如《生理学》《病理学》《传染病学》等）解析、构建知识图谱，并通过语义对齐算法完成了跨教材知识点整合。

# 你的职责
1. 向教师解释每一项整合决策的理由
2. 根据教师的反馈调整整合方案（保留/拆分/合并知识点）
3. 回答教师关于教材知识点的任何问题

# 知识问题回答规则（极其重要）
当上下文中包含「🔍 关键词检索结果」时，你必须严格按以下步骤回答：
- **第一步**：从检索结果中逐字摘录与问题直接相关的句子，放在 > 引用块中
- **第二步**：用你自己的话解释这段原文
- **禁止**：自己编造一句话然后贴上 [来源N] 标签——摘录必须是检索结果中真实存在的文字
- 如果检索结果中确实没有相关内容，回复"当前教材中未检索到相关信息"，然后可以补充自己的知识

# 整合决策回答规则
- 引用具体的教材名和知识点名
- 如果教师提出修改意见，先确认理解，再说明如何调整
- 回答控制在 200 字以内

# 无关话题
- 如果被问到无关话题（如天气、娱乐），礼貌拒绝并引导回正题

# 修改指令识别
当教师明确提出修改要求时，识别以下意图并在回复末尾附加 JSON：
- KEEP: 保留被删除的知识点
- SPLIT: 拆分被合并的知识点
- MERGE: 合并两个知识点
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

### 示例4：知识性问题（逐字引用检索结果）
教师：什么是静息电位？
（上下文包含检索结果：
[来源1] 《03_生理学》第二章 细胞的基本功能 p36:
"静息状态下细胞膜对K+的通透性最高，是因为细胞膜中存在持续开放的非门控钾通道...静息电位也就越接近该离子的平衡电位。")

AI：> "静息状态下细胞膜对K+的通透性最高…静息电位也就越接近该离子的平衡电位。" ——《03_生理学》第二章 p36

即静息电位主要由细胞膜对钾离子的高通透性决定，K⁺ 顺浓度差外流使膜内变负、膜外变正，最终接近 K⁺ 的平衡电位。

### 示例5：知识性问题（检索结果不足时）
教师：新冠肺炎的病理机制是什么？
（检索结果中无相关内容）

AI：当前教材中未检索到新冠肺炎相关信息。以下为个人知识补充：COVID-19 由 SARS-CoV-2 病毒引起，通过 ACE2 受体入侵细胞，引发细胞因子风暴导致肺组织损伤。如需详细机制请参考《传染病学》教材。

### 示例6：询问章节结构
教师：这本教材有哪些章节？帮我列出来
（上下文包含章节列表：
| ch_02 | 第一章 绪论 | 25 | 792字 |
| ch_22 | 第二章 细胞的基本功能 | 37 | 487字 | ...）

AI：## 《03_生理学》共 13 章

| 编号 | 标题 | 起始页 |
|------|------|--------|
| 第一章 | 绪论 | p25 |
| 第二章 | 细胞的基本功能 | p37 |
| 第三章 | 血液 | p77 |
| ... | ... | ... |

如需了解某章的具体内容，请告诉我章节编号。

### 示例7：询问某一章内容
教师：告诉我第一章讲了啥
（上下文包含第一章子节列表）

AI：第一章「绪论」涵盖以下内容：
- 第一节 生理学的研究对象和任务
- 第二节 生理学的常用研究方法
- 第三节 生命活动的基本特征（新陈代谢、兴奋性、适应性）
- ...

这一章是全书的基础，定义了生理学的研究范围和核心概念。

### 示例8：无关话题
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
    """AI 教师对话管理器 — 含关键词检索注入"""

    MAX_HISTORY = 12  # 保留最近 12 轮对话（24 条消息）

    def __init__(self):
        self.sessions: dict[str, DialogueSession] = {}
        self.retriever = KeywordRetriever()

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
    def _is_knowledge_question(user_msg: str) -> bool:
        """判断是否为知识性问题（需要检索教材内容）"""
        knowledge_patterns = [
            r'什么是', r'解释', r'说明', r'介绍一下', r'讲讲',
            r'定义', r'概念', r'原理', r'机制', r'功能',
            r'如何', r'怎么', r'为什么', r'区别', r'对比',
            r'关系', r'作用', r'特点', r'分类',
        ]
        decision_patterns = [
            r'为什么.*合并', r'为什么.*删除', r'保留', r'拆分',
            r'分开', r'整合决策', r'压缩',
        ]
        msg = user_msg.strip()
        if any(re.search(p, msg) for p in decision_patterns):
            return False
        return any(re.search(p, msg) for p in knowledge_patterns)

    @staticmethod
    def _is_chapter_query(user_msg: str) -> bool:
        """判断是否在问章节/目录结构"""
        chapter_patterns = [
            r'章节', r'目录', r'有哪些章', r'第.*章.*讲',
            r'讲了啥', r'讲了什么', r'标题', r'大纲',
            r'章.*内容', r'第.*章.*什么', r'一共.*章',
            r'几个章节', r'结构', r'列出.*章',
        ]
        return any(re.search(p, user_msg.strip()) for p in chapter_patterns)

    @staticmethod
    def _format_chapter_outline(books: dict) -> str:
        """将教材的全部章节标题格式化输出"""
        if not books:
            return "（暂无教材信息）"
        parts = []
        for bid, b in books.items():
            title = getattr(b, "title", str(b)[:30])
            chapters = getattr(b, "chapters", [])
            if not chapters:
                continue

            # 只列出主章节（level <= 2），过滤掉太细碎的子节
            main_chapters = [
                c for c in chapters
                if (hasattr(c, 'level') and c.level <= 2 and c.char_count > 50)
                or (isinstance(c, dict) and c.get('level', 99) <= 2 and c.get('char_count', 0) > 50)
            ]
            if not main_chapters:
                continue

            lines = [f"## 《{title}》共 {len(main_chapters)} 章\n"]
            lines.append("| 编号 | 章节标题 | 起始页 | 字数 |")
            lines.append("|------|----------|--------|------|")
            for ch in main_chapters:
                if hasattr(ch, 'chapter_id'):
                    cid, ctitle, lv, pg, cc = ch.chapter_id, ch.title, ch.level, ch.page_start, ch.char_count
                else:
                    cid = ch.get('chapter_id', '?')
                    ctitle = ch.get('title', '?')
                    lv = ch.get('level', 1)
                    pg = ch.get('page_start', 1)
                    cc = ch.get('char_count', 0)
                lines.append(f"| {cid} | {ctitle} | {pg} | {cc:,} |")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    @staticmethod
    def _format_chapter_detail(books: dict, chapter_num: int) -> str:
        """获取某一章的子节详情"""
        if not books:
            return ""
        parts = []
        for bid, b in books.items():
            title = getattr(b, "title", str(b)[:30])
            chapters = getattr(b, "chapters", [])

            # 匹配第X章的主章节
            import re as _re
            target_patterns = [
                _re.compile(rf'第\s*[一二三四五六七八九十\d]{{1,3}}\s*[章篇]'),
            ]
            main_chs = [
                (i, c) for i, c in enumerate(chapters)
                if any(p.match(c.title if hasattr(c, 'title') else c.get('title', ''))
                       for p in target_patterns)
                and (c.char_count if hasattr(c, 'char_count') else c.get('char_count', 0)) > 50
            ]

            if chapter_num <= len(main_chs):
                idx, main_ch = main_chs[chapter_num - 1]
                ch_title = main_ch.title if hasattr(main_ch, 'title') else main_ch.get('title', '')
                ch_page = main_ch.page_start if hasattr(main_ch, 'page_start') else main_ch.get('page_start', 1)

                # 收集该主章节下的子节
                parts.append(f"## 《{title}》{ch_title} (起始页: {ch_page})\n")
                next_main_idx = main_chs[chapter_num][0] if chapter_num < len(main_chs) else len(chapters)
                subs = chapters[idx + 1 : next_main_idx]
                # 只取前20个子节
                for s in subs[:20]:
                    st = s.title if hasattr(s, 'title') else s.get('title', '')
                    sp = s.page_start if hasattr(s, 'page_start') else s.get('page_start', 1)
                    sc = s.char_count if hasattr(s, 'char_count') else s.get('char_count', 0)
                    if sc > 20:
                        parts.append(f"- {st} (p{sp}, {sc}字)")
                if len(subs) > 20:
                    parts.append(f"... 共 {len(subs)} 个子节")
        return "\n".join(parts) if parts else ""

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
        active_books = state_books or books
        if active_books:
            context_parts.append(
                self._format_textbook_summary(active_books)
            )
            # 索引教材以便关键词检索
            self.retriever.index_books(active_books)

        # 整合决策摘要
        decisions_data = (
            session.integration_state.get("decisions") or decisions or []
        )
        if decisions_data:
            context_parts.append(self._format_decisions(decisions_data))

        # ── 章节目录查询：直接注入章节列表，不走关键词搜索 ──
        if active_books and self._is_chapter_query(user_msg):
            # 提取可能的章节编号
            chapter_num_match = re.search(r'第\s*([一二三四五六七八九十\d]+)\s*章', user_msg)
            if chapter_num_match:
                num_str = chapter_num_match.group(1)
                cn_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,
                          '十一':11,'十二':12,'十三':13,'十四':14,'十五':15}
                num = cn_map.get(num_str) or int(num_str) if num_str.isdigit() else None
                if num:
                    detail = self._format_chapter_detail(active_books, num)
                    if detail:
                        context_parts.append(detail)
            if not context_parts or len(context_parts) == 1:  # 只有教材摘要，没有章节详情
                outline = self._format_chapter_outline(active_books)
                if outline:
                    context_parts.append(outline)

        # ── 关键词检索：提取关键词 → 搜索教材 → 注入相关原文 ──
        elif active_books and self._is_knowledge_question(user_msg):
            retrieved = self.retriever.retrieve_and_format(user_msg, top_n=4)
            if retrieved:
                context_parts.append(retrieved)

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
