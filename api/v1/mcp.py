from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from backend.common.pagination import DependsPagination, PageData, paging_data
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.common.security.permission import RequestPermission
from backend.common.security.rbac import DependsRBAC
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.mcp import CreateMcpParam, GetMcpDetail, UpdateMcpParam
from backend.plugin.ai.service.mcp_service import mcp_service

router = APIRouter()


@router.get('/all', summary='获取所有 MCP', dependencies=[DependsJwtAuth])
async def get_all_mcps(db: CurrentSession) -> ResponseSchemaModel[list[GetMcpDetail]]:
    data = await mcp_service.get_all(db=db)
    return response_base.success(data=data)


@router.get('/{pk}', summary='获取 MCP 详情', dependencies=[DependsJwtAuth])
async def get_mcp(
    db: CurrentSession, pk: Annotated[int, Path(description='MCP ID')]
) -> ResponseSchemaModel[GetMcpDetail]:
    mcp = await mcp_service.get(db=db, pk=pk)
    return response_base.success(data=mcp)


@router.get(
    '',
    summary='分页获取所有 MCP',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_mcps_paginated(
    db: CurrentSession,
    name: Annotated[str | None, Query(description='MCP 名称')] = None,
    type: Annotated[int | None, Query(description='MCP 类型')] = None,
) -> ResponseSchemaModel[PageData[GetMcpDetail]]:
    mcp_select = await mcp_service.get_select(name=name, type=type)
    page_data = await paging_data(db, mcp_select)
    return response_base.success(data=page_data)


@router.post(
    '',
    summary='创建 MCP',
    dependencies=[
        Depends(RequestPermission('ai:mcp:add')),
        DependsRBAC,
    ],
)
async def create_mcp(db: CurrentSessionTransaction, obj: CreateMcpParam) -> ResponseModel:
    await mcp_service.create(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/{pk}',
    summary='更新 MCP',
    dependencies=[
        Depends(RequestPermission('ai:mcp:edit')),
        DependsRBAC,
    ],
)
async def update_mcp(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='MCP ID')], obj: UpdateMcpParam
) -> ResponseModel:
    count = await mcp_service.update(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/{pk}',
    summary='删除 MCP',
    dependencies=[
        Depends(RequestPermission('ai:mcp:del')),
        DependsRBAC,
    ],
)
async def delete_mcp(db: CurrentSessionTransaction, pk: Annotated[int, Path(description='MCP ID')]) -> ResponseModel:
    count = await mcp_service.delete(db=db, pk=pk)
    if count > 0:
        return response_base.success()
    return response_base.fail()
