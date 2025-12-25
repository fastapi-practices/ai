from typing import Annotated

from fastapi import APIRouter, Depends, Path

from backend.common.pagination import DependsPagination, PageData
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.common.security.permission import RequestPermission
from backend.common.security.rbac import DependsRBAC
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.model import (
    CreateAIModelParam,
    DeleteAIModelParam,
    GetAIModelDetail,
    UpdateAIModelParam,
)
from backend.plugin.ai.service.model_service import ai_model_service

router = APIRouter()


@router.get('/{pk}', summary='获取模型详情', dependencies=[DependsJwtAuth])
async def get_ai_model(
    db: CurrentSession, pk: Annotated[int, Path(description='模型 ID')]
) -> ResponseSchemaModel[GetAIModelDetail]:
    ai_model = await ai_model_service.get(db=db, pk=pk)
    return response_base.success(data=ai_model)


@router.get(
    '',
    summary='分页获取所有模型',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_ai_models_paginated(db: CurrentSession) -> ResponseSchemaModel[PageData[GetAIModelDetail]]:
    page_data = await ai_model_service.get_list(db=db)
    return response_base.success(data=page_data)


@router.post(
    '',
    summary='创建模型',
    dependencies=[
        Depends(RequestPermission('ai:model:add')),
        DependsRBAC,
    ],
)
async def create_ai_model(db: CurrentSessionTransaction, obj: CreateAIModelParam) -> ResponseModel:
    await ai_model_service.create(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/{pk}',
    summary='更新模型',
    dependencies=[
        Depends(RequestPermission('ai:model:edit')),
        DependsRBAC,
    ],
)
async def update_ai_model(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='模型 ID')], obj: UpdateAIModelParam
) -> ResponseModel:
    count = await ai_model_service.update(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '',
    summary='批量删除模型',
    dependencies=[
        Depends(RequestPermission('ai:model:del')),
        DependsRBAC,
    ],
)
async def delete_ai_models(db: CurrentSessionTransaction, obj: DeleteAIModelParam) -> ResponseModel:
    count = await ai_model_service.delete(db=db, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()
