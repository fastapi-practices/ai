from fastapi import APIRouter

from backend.core.conf import settings
from backend.plugin.ai.api.v1.chat import router as chat_router
from backend.plugin.ai.api.v1.model import router as model_router
from backend.plugin.ai.api.v1.provider import router as provider_router

v1 = APIRouter(prefix=settings.FASTAPI_API_V1_PATH)

v1.include_router(chat_router, prefix='/chat', tags=['AI 文本生成'])
v1.include_router(model_router, prefix='/models', tags=['AI 模型管理'])
v1.include_router(provider_router, prefix='/providers', tags=['AI 供应商管理'])
