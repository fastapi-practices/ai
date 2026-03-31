from typing import Annotated

from fastapi import APIRouter, Path, Request
from fastapi.responses import Response

from backend.common.pagination import CursorPageData, DependsCursorPagination
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.chat import AIChatRegenerateParam
from backend.plugin.ai.schema.conversation import (
    ClearAIConversationContextResult,
    GetAIConversationDetail,
    GetAIConversationListItem,
    UpdateAIConversationPinnedParam,
    UpdateAIConversationTitleParam,
)
from backend.plugin.ai.schema.message import UpdateAIMessageParam
from backend.plugin.ai.service.chat_service import ai_chat_service
from backend.plugin.ai.service.conversation_service import ai_conversation_service

router = APIRouter()


@router.get('/{pk}', summary='获取对话详情', dependencies=[DependsJwtAuth])
async def get_conversation(
    request: Request,
    db: CurrentSession,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseSchemaModel[GetAIConversationDetail]:
    data = await ai_conversation_service.get(db=db, conversation_id=pk, user_id=request.user.id)
    return response_base.success(data=data)


@router.get(
    '',
    summary='分页获取对话列表',
    dependencies=[
        DependsJwtAuth,
        DependsCursorPagination,
    ],
)
async def get_conversations_paginated(
    request: Request,
    db: CurrentSession,
) -> ResponseSchemaModel[CursorPageData[GetAIConversationListItem]]:
    data = await ai_conversation_service.get_list(db=db, user_id=request.user.id)
    return response_base.success(data=data)


@router.put('/{pk}', summary='更新对话标题', dependencies=[DependsJwtAuth])
async def update_conversation(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    obj: UpdateAIConversationTitleParam,
) -> ResponseModel:
    count = await ai_conversation_service.update(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.put('/{pk}/pin', summary='更新对话置顶状态', dependencies=[DependsJwtAuth])
async def update_conversation_pinned_status(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    obj: UpdateAIConversationPinnedParam,
) -> ResponseModel:
    count = await ai_conversation_service.update_pinned_status(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete('/{pk}', summary='删除对话', dependencies=[DependsJwtAuth])
async def delete_conversation(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseModel:
    count = await ai_conversation_service.delete(db=db, conversation_id=pk, user_id=request.user.id)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/{pk}/messages',
    summary='清空对话消息',
    dependencies=[DependsJwtAuth],
)
async def clear_conversation_messages(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseModel:
    count = await ai_conversation_service.clear_messages(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/{pk}/messages/{message_id}',
    summary='删除指定消息',
    dependencies=[DependsJwtAuth],
)
async def delete_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
) -> ResponseModel:
    count = await ai_conversation_service.delete_message(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        message_id=message_id,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.put(
    '/{pk}/messages/{message_id}',
    summary='编辑保存指定消息',
    dependencies=[DependsJwtAuth],
)
async def update_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: UpdateAIMessageParam,
) -> ResponseModel:
    count = await ai_conversation_service.update_message(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        message_id=message_id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.post(
    '/{pk}/messages/{message_id}/regenerate',
    summary='根据用户消息重生成 AI 回复',
    dependencies=[DependsJwtAuth],
)
async def regenerate_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: AIChatRegenerateParam,
) -> Response:
    return await ai_chat_service.regenerate_from_user_message(
        db=db,
        user_id=request.user.id,
        conversation_id=pk,
        message_id=message_id,
        obj=obj,
        accept=request.headers.get('accept'),
    )


@router.post(
    '/{pk}/responses/{message_id}/regenerate',
    summary='根据 AI 回复重生成',
    dependencies=[DependsJwtAuth],
)
async def regenerate_conversation_response(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: AIChatRegenerateParam,
) -> Response:
    return await ai_chat_service.regenerate_from_response_message(
        db=db,
        user_id=request.user.id,
        conversation_id=pk,
        message_id=message_id,
        obj=obj,
        accept=request.headers.get('accept'),
    )


@router.post(
    '/{pk}/clear-context',
    summary='清除对话上下文',
    dependencies=[DependsJwtAuth],
)
async def clear_conversation_context(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseSchemaModel[ClearAIConversationContextResult]:
    data = await ai_conversation_service.clear_context(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
    )
    return response_base.success(data=data)
