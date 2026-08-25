from fastapi import APIRouter, Request
from pydantic_ai.ui import SSE_CONTENT_TYPE
from starlette.responses import StreamingResponse

from backend.common.security.jwt import DependsJwtAuth
from backend.plugin.ai.schema.chat import AIChatCompletionParam
from backend.plugin.ai.service.chat_service import ai_chat_service

router = APIRouter()


@router.post(
    '/completions',
    summary='流式生成',
    dependencies=[DependsJwtAuth],
)
async def create_ai_chat_completion(
    request: Request,
    obj: AIChatCompletionParam,
) -> StreamingResponse:
    return await ai_chat_service.create_completion(
        user_id=request.user.id,
        obj=obj,
        accept=request.headers.get('accept', SSE_CONTENT_TYPE),
    )
