"""
LLM Chat Assistant — FastAPI + OpenAI 最小可运行示例

启动（先激活环境：conda activate llm-chat-assistant）：
    python -m uvicorn main:app --reload

然后浏览器打开 http://127.0.0.1:8000/docs
FastAPI 会自动生成交互式 API 文档，可以直接在页面里测试接口。
"""

import os

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------
# 1) 读取 .env 里的配置（如 OPENAI_API_KEY）
#    项目里没有 .env 时也不报错，只是 API Key 取不到
# ---------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------
# 2) 创建 FastAPI 应用实例
#    app 是一个 ASGI 应用对象，uvicorn main:app 加载的就是它
# ---------------------------------------------------------------
app = FastAPI(title="LLM Chat Assistant", version="0.1.0")

# ---------------------------------------------------------------
# 3) OpenAI 客户端：懒加载，第一次真正调用接口时才创建
#    （如果一开始就 openai.OpenAI()，没有密钥会直接抛错导致程序起不来）
# ---------------------------------------------------------------
_client: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="未设置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填入你的密钥",
            )
        _client = openai.OpenAI()
    return _client


# ---------------------------------------------------------------
# 4) 用 Pydantic 定义请求 / 响应模型
#    FastAPI 会根据它们自动做参数校验、类型转换，并生成接口文档
# ---------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str                                    # 必填：用户消息
    system_prompt: str = "你是一个乐于助人的AI助手。"   # 可选：系统提示词
    max_tokens: int = 512                           # 可选：回复最大 token 数


class ChatResponse(BaseModel):
    reply: str
    model: str


# ---------------------------------------------------------------
# 5) 定义接口（路由）
# ---------------------------------------------------------------
@app.get("/")
def read_root():
    """GET / —— 健康检查，确认服务在运行"""
    return {"message": "Hello, I'm the LLM Chat Assistant. Try POST /chat"}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    """POST /chat —— 对话接口：把消息转发给 OpenAI 并返回回复"""
    try:
        completion = get_client().chat.completions.create(
            model="gpt-4o-mini",       # 可换成 gpt-4o / 其它兼容模型
            messages=[
                {"role": "system", "content": chat_request.system_prompt},
                {"role": "user", "content": chat_request.message},
            ],
            max_tokens=chat_request.max_tokens,
        )
        return ChatResponse(
            reply=completion.choices[0].message.content or "",
            model=completion.model,
        )
    except openai.OpenAIError as exc:
        # 网络错误 / 密钥无效 / 余额不足等，统一返回 HTTP 502
        raise HTTPException(status_code=502, detail=f"OpenAI 调用失败：{exc}")
