"""
共享 LLM 客户端 — 千问3-0.6B (ModelScope)
对话模块 & RAG 模块统一通过此客户端调用大模型
"""

import os
from openai import OpenAI

# 加载 .env（确保在任何导入路径下都能读到配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Qwen3 ModelScope 配置 ──────────────────────
QWEN_BASE_URL = "https://ms-ens-f8274faf-bcde.api-inference.modelscope.cn/v1"
QWEN_API_KEY = "ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2"
QWEN_MODEL = "Qwen/Qwen3-0.6B"

# 环境变量可覆盖（用 MY_LLM_* 前缀避免和 ModelScope 平台默认值冲突）
BASE_URL = os.getenv("MY_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or QWEN_BASE_URL
API_KEY = os.getenv("MY_LLM_API_KEY") or os.getenv("LLM_API_KEY") or QWEN_API_KEY
MODEL = os.getenv("MY_LLM_MODEL") or os.getenv("LLM_MODEL") or QWEN_MODEL

# 如果检测到被 ModelScope 注入了 api.openai.com，强制用 Qwen3
if "api.openai.com" in BASE_URL:
    print(f"[LLM Client] ⚠️ 检测到 ModelScope 默认 LLM_BASE_URL，强制使用 Qwen3")
    BASE_URL = QWEN_BASE_URL
    API_KEY = QWEN_API_KEY
    MODEL = QWEN_MODEL

# 启动诊断：打印实际使用的 LLM 配置（API Key 中间脱敏）
def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "****" + key[-4:]
    return key[:6] + "*" * (len(key) - 10) + key[-4:]

print(f"[LLM Client] BASE_URL = {BASE_URL}")
print(f"[LLM Client] MODEL    = {MODEL}")
print(f"[LLM Client] API_KEY  = {_mask_key(API_KEY)}")

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
        if not full_text:
            return f"[LLM 返回为空] BASE_URL={BASE_URL} MODEL={MODEL} API_KEY前缀={API_KEY[:15]}..."
        return full_text
    except Exception as e:
        import traceback
        detail = traceback.format_exc()
        # 截取关键错误信息
        short = str(e)[:300]
        print(f"[LLM ERROR] {short}")
        print(f"[LLM TRACE] {detail[:500]}")
        return f"[LLM 调用失败] {type(e).__name__}: {short}"


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
