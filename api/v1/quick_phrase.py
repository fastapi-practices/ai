from typing import Annotated

from fastapi import APIRouter, Path, Query, Request

from backend.common.pagination import DependsPagination, PageData
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.quick_phrase import (
    CreateAIQuickPhraseParam,
    GetAIQuickPhraseDetail,
    UpdateAIQuickPhraseParam,
)
from backend.plugin.ai.service.quick_phrase_service import ai_quick_phrase_service

router = APIRouter()


@router.get('/all', summary='获取快捷短语列表', dependencies=[DependsJwtAuth])
async def get_ai_quick_phrases(
    request: Request, db: CurrentSession
) -> ResponseSchemaModel[list[GetAIQuickPhraseDetail]]:
    data = await ai_quick_phrase_service.get_all(db=db, user_id=request.user.id)
    return response_base.success(data=data)


@router.get('/{pk}', summary='获取快捷短语详情', dependencies=[DependsJwtAuth])
async def get_ai_quick_phrase(
    request: Request,
    db: CurrentSession,
    pk: Annotated[int, Path(description='快捷短语 ID')],
) -> ResponseSchemaModel[GetAIQuickPhraseDetail]:
    data = await ai_quick_phrase_service.get(db=db, pk=pk, user_id=request.user.id)
    return response_base.success(data=data)


@router.get(
    '',
    summary='分页获取快捷短语',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_ai_quick_phrases_paginated(
    request: Request,
    db: CurrentSession,
    content: Annotated[str | None, Query(description='短语内容')] = None,
) -> ResponseSchemaModel[PageData[GetAIQuickPhraseDetail]]:
    page_data = await ai_quick_phrase_service.get_list(db=db, user_id=request.user.id, content=content)
    return response_base.success(data=page_data)


@router.post('', summary='创建快捷短语', dependencies=[DependsJwtAuth])
async def create_ai_quick_phrase(
    request: Request,
    db: CurrentSessionTransaction,
    obj: CreateAIQuickPhraseParam,
) -> ResponseModel:
    await ai_quick_phrase_service.create(db=db, obj=obj, user_id=request.user.id)
    return response_base.success()


@router.put('/{pk}', summary='更新快捷短语', dependencies=[DependsJwtAuth])
async def update_ai_quick_phrase(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[int, Path(description='快捷短语 ID')],
    obj: UpdateAIQuickPhraseParam,
) -> ResponseModel:
    count = await ai_quick_phrase_service.update(db=db, pk=pk, obj=obj, user_id=request.user.id)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete('/{pk}', summary='删除快捷短语', dependencies=[DependsJwtAuth])
async def delete_ai_quick_phrase(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[int, Path(description='快捷短语 ID')],
) -> ResponseModel:
    count = await ai_quick_phrase_service.delete(db=db, pk=pk, user_id=request.user.id)
    if count > 0:
        return response_base.success()
    return response_base.fail()
