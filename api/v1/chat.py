from typing import Annotated

from fastapi import APIRouter, Body, Path, Request
from fastapi.sse import EventSourceResponse

from backend.common.pagination import CursorPageData, DependsCursorPagination
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.chat import AIChatParam, UpdateAIChatMessageParam
from backend.plugin.ai.schema.chat_history import (
    DeleteAIChatMessageResult,
    GetAIChatConversationDetail,
    GetAIChatConversationListItem,
    UpdateAIChatConversationParam,
    UpdateAIChatConversationPinParam,
)
from backend.plugin.ai.service.chat_history_service import ai_chat_history_service
from backend.plugin.ai.service.chat_service import ai_chat_service

router = APIRouter()


@router.post(
    '/completions',
    summary='文本生成（对话）',
    description=(
        '统一聊天入口，支持三种请求模式：'
        '1. 普通发送：传 `user_prompt`，`conversation_id` 可不传'
        '2. 编辑重发：传 `conversation_id`、`edit_message_id`、`user_prompt`'
        '3. 重新生成：传 `conversation_id`、`regenerate_message_id`，无需再传 `user_prompt`'
    ),
    dependencies=[DependsJwtAuth],
)
async def create_ai_chat_completion(
    request: Request,
    db: CurrentSessionTransaction,
    chat: Annotated[AIChatParam, Body()],
) -> EventSourceResponse:
    return EventSourceResponse(
        ai_chat_service.stream_messages(db=db, chat=chat, user_id=request.user.id),
    )


@router.get(
    '/conversations',
    summary='分页获取聊天话题',
    dependencies=[
        DependsJwtAuth,
        DependsCursorPagination,
    ],
)
async def get_ai_chat_conversations_paginated(
    request: Request,
    db: CurrentSession,
) -> ResponseSchemaModel[CursorPageData[GetAIChatConversationListItem]]:
    data = await ai_chat_history_service.get_list(db=db, user_id=request.user.id)
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
async def update_ai_chat_conversation_pinned_status(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    obj: UpdateAIChatConversationPinParam,
) -> ResponseModel:
    count = await ai_chat_history_service.update_pinned_status(
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
async def clear_ai_chat_conversation_messages(
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
    '/conversations/{conversation_id}/messages/{message_id}',
    summary='删除指定聊天消息',
    dependencies=[DependsJwtAuth],
)
async def delete_ai_chat_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
) -> ResponseSchemaModel[DeleteAIChatMessageResult]:
    data = await ai_chat_history_service.delete_message(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
        message_id=message_id,
    )
    return response_base.success(data=data)


@router.put(
    '/conversations/{conversation_id}/messages/{message_id}',
    summary='编辑保存指定用户消息',
    dependencies=[DependsJwtAuth],
)
async def update_ai_chat_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    conversation_id: Annotated[str, Path(description='会话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: UpdateAIChatMessageParam,
) -> ResponseModel:
    count = await ai_chat_history_service.update_message(
        db=db,
        conversation_id=conversation_id,
        user_id=request.user.id,
        message_id=message_id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()
