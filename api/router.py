from fastapi import APIRouter

from backend.core.conf import settings
from backend.plugin.ai.api.v1.chat import router as chat_router
from backend.plugin.ai.api.v1.conversation import router as conversation_router
from backend.plugin.ai.api.v1.default_model import router as default_model_router
from backend.plugin.ai.api.v1.message import router as message_router
from backend.plugin.ai.api.v1.model import router as model_router
from backend.plugin.ai.api.v1.model_option import router as model_option_router
from backend.plugin.ai.api.v1.provider import router as provider_router
from backend.plugin.ai.api.v1.quick_phrase import router as quick_phrase_router

v1 = APIRouter(prefix=settings.FASTAPI_API_V1_PATH)

v1.include_router(chat_router, prefix='/chat', tags=['AI 生成'])
v1.include_router(model_option_router, prefix='/model-options', tags=['AI 模型管理'])
v1.include_router(default_model_router, prefix='/default-models', tags=['AI 默认模型管理'])
v1.include_router(conversation_router, prefix='/conversations', tags=['AI 对话管理'])
v1.include_router(message_router, prefix='/conversations', tags=['AI 消息管理'])
v1.include_router(model_router, prefix='/models', tags=['AI 模型管理'])
v1.include_router(provider_router, prefix='/providers', tags=['AI 供应商管理'])
v1.include_router(quick_phrase_router, prefix='/quick-phrases', tags=['AI 快捷短语'])
