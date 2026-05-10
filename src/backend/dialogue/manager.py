"""
多轮对话管理模块
支持教师通过对话查看和修改整合决策
"""

import os
from dataclasses import dataclass, field
from llm_client import call_llm


@dataclass
class Message:
    role: str
    content: str


@dataclass
class DialogueSession:
    session_id: str
    messages: list = field(default_factory=list)
    integration_state: dict = field(default_factory=dict)

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def get_history(self) -> list:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class DialogueManager:
    def __init__(self):
        self.sessions: dict[str, DialogueSession] = {}

    def get_or_create(self, sid: str) -> DialogueSession:
        if sid not in self.sessions:
            s = DialogueSession(session_id=sid)
            s.add("system", "你是一位学科教学专家。你可以向教师解释整合决策的理由，并根据反馈调整方案。请用中文回答。")
            self.sessions[sid] = s
        return self.sessions[sid]

    def chat(self, sid: str, user_msg: str, decisions: list = None) -> str:
        session = self.get_or_create(sid)
        if decisions:
            session.integration_state = {"decisions": [d.to_dict() if hasattr(d, 'to_dict') else d for d in decisions]}
        session.add("user", user_msg)

        # 构建上下文
        ctx = ""
        if session.integration_state:
            ctx = f"\n当前整合决策:\n{session.integration_state}"

        system = session.messages[0].content if session.messages else "你是一位学科教学专家。"
        prompt = f"""对话历史：
{chr(10).join(f'{m.role}: {m.content}' for m in session.messages[1:-1])}

{ctx}

用户的问题/反馈：
{user_msg}

请回复："""

        reply = call_llm(prompt, system, temperature=0.5)
        session.add("assistant", reply)
        return reply
