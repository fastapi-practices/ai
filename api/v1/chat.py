from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query, Request
from starlette.responses import StreamingResponse

from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.chat import AIChatParam
from backend.plugin.ai.schema.chat_history import (
    DeleteAIChatMessageResult,
    GetAIChatConversationDetail,
    GetAIChatConversationList,
    UpdateAIChatConversationParam,
    UpdateAIChatConversationPinParam,
)
from backend.plugin.ai.service.chat_history_service import ai_chat_history_service
from backend.plugin.ai.service.chat_service import ai_chat_service

router = APIRouter()


@router.post('/completions', summary='文本生成（对话）', dependencies=[DependsJwtAuth])
async def completions(
    request: Request,
    db: CurrentSessionTransaction,
    chat: AIChatParam,
) -> StreamingResponse:
    return StreamingResponse(
        ai_chat_service.stream_messages(db=db, chat=chat, user_id=request.user.id),
        media_type='application/x-ndjson',
    )


@router.get('/conversations', summary='获取最近聊天历史', dependencies=[DependsJwtAuth])
async def get_ai_chat_conversations(
    request: Request,
    db: CurrentSession,
    limit: Annotated[int, Query(ge=1, le=100, description='返回数量')] = 20,
    before: Annotated[datetime | None, Query(description='查询游标，取该时间之前的会话')] = None,
) -> ResponseSchemaModel[GetAIChatConversationList]:
    data = await ai_chat_history_service.get_recent_list(db=db, user_id=request.user.id, limit=limit, before=before)
    return response_base.success(data=data)


@router.get('/conversations/{conversation_id}', summary='获取聊天历史详情', dependencies=[DependsJwtAuth])
async def get_ai_chat_conversation(
    request: Request,
    db: CurrentSession,
    conversation_id: Annotated[str, Path(description='会话 ID')],
) -> ResponseSchemaModel[GetAIChatConversationDetail]:
    data = await ai_chat_history_service.get_detail(db=db, conversation_id=conversation_id, user_id=request.user.id)
    return response_base.success(data=data)


@router.put('/conversations/{conversation_id}', summary='更新聊天话题', dependencies=[DependsJwtAuth])
async def update_ai_chat_conversation(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    obj: UpdateAIChatConversationParam,
) -> ResponseModel:
    count = await ai_chat_history_service.update(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.put('/conversations/{conversation_id}/pin', summary='置顶聊天话题', dependencies=[DependsJwtAuth])
async def update_ai_chat_conversation_pin(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    obj: UpdateAIChatConversationPinParam,
) -> ResponseModel:
    count = await ai_chat_history_service.update_pin(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete('/conversations/{conversation_id}', summary='删除聊天历史', dependencies=[DependsJwtAuth])
async def delete_ai_chat_conversation(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
) -> ResponseModel:
    count = await ai_chat_history_service.delete(db=db, conversation_id=conversation_id, user_id=request.user.id)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/conversations/{conversation_id}/messages',
    summary='清空话题对话历史',
    dependencies=[DependsJwtAuth],
)
async def clear_ai_chat_messages(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
) -> ResponseModel:
    count = await ai_chat_history_service.clear_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/conversations/{conversation_id}/messages/{message_index}',
    summary='删除指定聊天消息',
    dependencies=[DependsJwtAuth],
)
async def delete_ai_chat_message(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    message_index: Annotated[int, Path(ge=0, description='消息索引')],
) -> ResponseSchemaModel[DeleteAIChatMessageResult]:
    data = await ai_chat_history_service.delete_message(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
        message_index=message_index,
    )
    return response_base.success(data=data)
