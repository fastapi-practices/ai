from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic_ai.ui import SSE_CONTENT_TYPE

from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSessionTransaction
from backend.plugin.ai.service.chat_service import ai_chat_service

router = APIRouter()


@router.post(
    '/completions',
    summary='流式生成',
    dependencies=[DependsJwtAuth],
)
async def create_ai_chat_completion(
    request: Request,
    db: CurrentSessionTransaction,
) -> Response:
    return await ai_chat_service.create_completion(
        db=db,
        user_id=request.user.id,
        body=await request.body(),
        accept=request.headers.get('accept', SSE_CONTENT_TYPE),
    )
