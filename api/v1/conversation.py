from typing import Annotated

from fastapi import APIRouter, Path, Request

from backend.common.pagination import CursorPageData, DependsCursorPagination
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.conversation import (
    GetAIConversationDetail,
    GetAIConversationListItem,
    UpdateAIConversationPinnedParam,
    UpdateAIConversationTitleParam,
)
from backend.plugin.ai.service.conversation_service import ai_conversation_service

router = APIRouter()


@router.get(
    '/{pk}',
    summary='获取对话详情',
    dependencies=[DependsJwtAuth],
)
async def get_conversation(
    request: Request,
    db: CurrentSession,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseSchemaModel[GetAIConversationDetail]:
    data = await ai_conversation_service.get(db=db, conversation_id=pk, user_id=request.user.id)
    return response_base.success(data=data)


@router.get(
    '',
    summary='分页获取所有对话列表',
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


@router.post(
    '/{pk}/clear-context',
    summary='清除对话上下文',
    dependencies=[DependsJwtAuth],
)
async def clear_conversation_context(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseModel:
    count = await ai_conversation_service.clear_context(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
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
