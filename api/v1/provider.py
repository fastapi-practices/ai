from typing import Annotated

from fastapi import APIRouter, Depends, Path

from backend.common.pagination import DependsPagination, PageData
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.common.security.permission import RequestPermission
from backend.common.security.rbac import DependsRBAC
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.provider import (
    CreateAiProviderParam,
    DeleteAiProviderParam,
    GetAiProviderDetail,
    UpdateAiProviderParam,
)
from backend.plugin.ai.service.provider_service import ai_provider_service

router = APIRouter()


@router.get('/{pk}', summary='获取供应商详情', dependencies=[DependsJwtAuth])
async def get_ai_provider(
    db: CurrentSession, pk: Annotated[int, Path(description='provider ID')]
) -> ResponseSchemaModel[GetAiProviderDetail]:
    ai_provider = await ai_provider_service.get(db=db, pk=pk)
    return response_base.success(data=ai_provider)


@router.get(
    '',
    summary='分页获取所有供应商',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_ai_providers_paginated(db: CurrentSession) -> ResponseSchemaModel[PageData[GetAiProviderDetail]]:
    page_data = await ai_provider_service.get_list(db=db)
    return response_base.success(data=page_data)


@router.post(
    '',
    summary='创建provider',
    dependencies=[
        Depends(RequestPermission('ai:provider:add')),
        DependsRBAC,
    ],
)
async def create_ai_provider(db: CurrentSessionTransaction, obj: CreateAiProviderParam) -> ResponseModel:
    await ai_provider_service.create(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/{pk}',
    summary='更新供应商',
    dependencies=[
        Depends(RequestPermission('ai:provider:edit')),
        DependsRBAC,
    ],
)
async def update_ai_provider(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='供应商 ID')], obj: UpdateAiProviderParam
) -> ResponseModel:
    count = await ai_provider_service.update(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '',
    summary='批量删除供应商',
    dependencies=[
        Depends(RequestPermission('ai:provider:del')),
        DependsRBAC,
    ],
)
async def delete_ai_providers(db: CurrentSessionTransaction, obj: DeleteAiProviderParam) -> ResponseModel:
    count = await ai_provider_service.delete(db=db, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()
