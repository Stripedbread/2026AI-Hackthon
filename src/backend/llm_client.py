"""
共享 LLM 客户端 — 千问3-0.6B (ModelScope)
对话模块 & RAG 模块统一通过此客户端调用大模型
"""

import os
from openai import OpenAI

# ── Qwen3 ModelScope 配置 ──────────────────────
QWEN_BASE_URL = "https://ms-ens-f8274faf-bcde.api-inference.modelscope.cn/v1"
QWEN_API_KEY = "ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2"
QWEN_MODEL = "Qwen/Qwen3-0.6B"

# 兼容旧的环境变量配置（优先级：环境变量 > 默认 Qwen3）
BASE_URL = os.getenv("LLM_BASE_URL", QWEN_BASE_URL)
API_KEY = os.getenv("LLM_API_KEY", QWEN_API_KEY)
MODEL = os.getenv("LLM_MODEL", QWEN_MODEL)

_client: OpenAI = None


def get_client() -> OpenAI:
    """获取全局 OpenAI 客户端（懒加载单例）"""
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def chat(messages: list[dict], temperature: float = 0.5, max_tokens: int = 2048) -> str:
    """调用 LLM 进行对话（内部使用流式，因为 ModelScope Qwen3 仅支持流式），返回完整回复文本"""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,  # Qwen3 ModelScope 端点仅支持流式
        )
        full_text = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
        return full_text or "[LLM 返回为空]"
    except Exception as e:
        return f"[LLM 调用失败: {e}]"


def chat_stream(messages: list[dict], temperature: float = 0.5, max_tokens: int = 2048):
    """调用 LLM 进行流式对话，返回生成器 yield 每个 token"""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
    except Exception as e:
        yield f"[LLM 调用失败: {e}]"


def call_llm(prompt: str, system: str = "", temperature: float = 0.5) -> str:
    """便捷函数：system + user prompt → 完整回复"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature)
