from fastapi import APIRouter

from backend.core.conf import settings
from backend.plugin.ai.api.v1.chat import router as chat_router

v1 = APIRouter(prefix=settings.FASTAPI_API_V1_PATH)

v1.include_router(chat_router, prefix='/chat', tags=['AI 文本生成'])
