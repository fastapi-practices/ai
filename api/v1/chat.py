from fastapi import APIRouter
from starlette.responses import StreamingResponse

from backend.database.db import CurrentSession
from backend.plugin.ai.schema.chat import AIChat
from backend.plugin.ai.service.chat_service import ai_chat_service

router = APIRouter()


@router.post('/completions', summary='文本生成（对话）')
async def completions(db: CurrentSession, chat: AIChat) -> StreamingResponse:
    return StreamingResponse(ai_chat_service.stream_messages(db=db, chat=chat), media_type='text/plain')
