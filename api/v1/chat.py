from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter()


@router.post('/completions', summary='文本生成（对话）')
async def completions() -> StreamingResponse: ...
