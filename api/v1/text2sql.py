from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request

from backend.common.pagination import DependsPagination, PageData
from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.common.security.permission import RequestPermission
from backend.common.security.rbac import DependsRBAC
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.schema.text2sql import (
    CreateText2SqlDatasetParam,
    CreateText2SqlExampleParam,
    CreateText2SqlTableParam,
    GetText2SqlDatasetDetail,
    GetText2SqlExampleDetail,
    GetText2SqlTableDetail,
    Text2SqlDatasetEnabled,
    Text2SqlQueryParam,
    Text2SqlQueryResult,
    Text2SqlTableSelectable,
    UpdateText2SqlDatasetParam,
    UpdateText2SqlExampleParam,
    UpdateText2SqlTableParam,
)
from backend.plugin.ai.service.text2sql_service import text2sql_service
from backend.plugin.ai.text2sql.engine import run_query

router = APIRouter()


# ---------------- 数据集 ----------------


@router.get('/datasets/enabled', summary='获取启用的数据集（chat 选择器）', dependencies=[DependsJwtAuth])
async def get_enabled_datasets(db: CurrentSession) -> ResponseSchemaModel[list[Text2SqlDatasetEnabled]]:
    data = await text2sql_service.get_enabled_datasets(db=db)
    return response_base.success(data=data)


@router.get('/datasets/all', summary='获取全部数据集', dependencies=[DependsJwtAuth])
async def get_all_datasets(db: CurrentSession) -> ResponseSchemaModel[list[GetText2SqlDatasetDetail]]:
    data = await text2sql_service.get_all_datasets(db=db)
    return response_base.success(data=data)


@router.get('/datasets/{pk}', summary='获取数据集详情', dependencies=[DependsJwtAuth])
async def get_dataset(
    db: CurrentSession, pk: Annotated[int, Path(description='数据集 ID')]
) -> ResponseSchemaModel[GetText2SqlDatasetDetail]:
    data = await text2sql_service.get_dataset(db=db, pk=pk)
    return response_base.success(data=data)


@router.get(
    '/datasets',
    summary='分页获取数据集',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_datasets(
    db: CurrentSession,
    name: Annotated[str | None, Query(description='数据集名称（模糊）')] = None,
    enabled: Annotated[int | None, Query(description='是否启用（0停用 1启用）')] = None,
) -> ResponseSchemaModel[PageData[GetText2SqlDatasetDetail]]:
    page_data = await text2sql_service.get_dataset_list(db=db, name=name, enabled=enabled)
    return response_base.success(data=page_data)


@router.post(
    '/datasets',
    summary='新增数据集',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:dataset:add')),
        DependsRBAC,
    ],
)
async def create_dataset(db: CurrentSessionTransaction, obj: CreateText2SqlDatasetParam) -> ResponseModel:
    await text2sql_service.create_dataset(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/datasets/{pk}',
    summary='更新数据集',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:dataset:edit')),
        DependsRBAC,
    ],
)
async def update_dataset(
    db: CurrentSessionTransaction,
    pk: Annotated[int, Path(description='数据集 ID')],
    obj: UpdateText2SqlDatasetParam,
) -> ResponseModel:
    count = await text2sql_service.update_dataset(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/datasets/{pk}',
    summary='删除数据集',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:dataset:del')),
        DependsRBAC,
    ],
)
async def delete_dataset(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='数据集 ID')]
) -> ResponseModel:
    count = await text2sql_service.delete_dataset(db=db, pk=pk)
    if count > 0:
        return response_base.success()
    return response_base.fail()


# ---------------- 自然语言查询 ----------------


@router.post(
    '/queries',
    summary='自然语言查询（Text2SQL）',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:query')),
        DependsRBAC,
    ],
)
async def text2sql_query(
    request: Request,
    db: CurrentSession,
    obj: Text2SqlQueryParam,
) -> ResponseSchemaModel[Text2SqlQueryResult]:
    tables = await text2sql_service.get_enabled(db=db, dataset_id=obj.dataset_id)
    examples = await text2sql_service.get_examples_for(
        db=db,
        tables={table.table_name for table in tables},
        dataset_id=obj.dataset_id,
    )
    data = await run_query(
        db=db,
        question=obj.question,
        user_id=request.user.id,
        selected_tables=tables,
        examples=examples,
    )
    return response_base.success(data=data)


# ---------------- 数据源管理（已选表） ----------------


@router.get('/tables', summary='获取可挑选的数据库表', dependencies=[DependsJwtAuth])
async def get_selectable_tables(
    db: CurrentSession,
    dataset_id: Annotated[int, Query(description='所属数据集 ID')],
    table_schema: Annotated[str | None, Query(description='库名/schema，缺省取 AI_TEXT2SQL_SCHEMA')] = None,
) -> ResponseSchemaModel[list[Text2SqlTableSelectable]]:
    data = await text2sql_service.list_selectable_tables(db=db, dataset_id=dataset_id, table_schema=table_schema)
    return response_base.success(data=data)


@router.get('/tables/{table_name}/columns', summary='获取表列信息', dependencies=[DependsJwtAuth])
async def get_table_columns(
    db: CurrentSession,
    table_name: Annotated[str, Path(description='表名')],
    table_schema: Annotated[str | None, Query(description='库名/schema')] = None,
) -> ResponseSchemaModel[list[dict[str, Any]]]:
    data = await text2sql_service.get_table_columns(db=db, table_name=table_name, table_schema=table_schema)
    return response_base.success(data=data)


@router.get('/selected-tables/all', summary='获取全部已选表', dependencies=[DependsJwtAuth])
async def get_all_selected_tables(db: CurrentSession) -> ResponseSchemaModel[list[GetText2SqlTableDetail]]:
    data = await text2sql_service.get_all_selected(db=db)
    return response_base.success(data=data)


@router.get('/selected-tables/{pk}', summary='获取已选表详情', dependencies=[DependsJwtAuth])
async def get_selected_table(
    db: CurrentSession, pk: Annotated[int, Path(description='已选表 ID')]
) -> ResponseSchemaModel[GetText2SqlTableDetail]:
    data = await text2sql_service.get_selected(db=db, pk=pk)
    return response_base.success(data=data)


@router.get(
    '/selected-tables',
    summary='分页获取已选表',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_selected_tables(
    db: CurrentSession,
    dataset_id: Annotated[int | None, Query(description='所属数据集 ID')] = None,
    schema_name: Annotated[str | None, Query(description='库名/schema')] = None,
    table_name: Annotated[str | None, Query(description='表名')] = None,
    enabled: Annotated[int | None, Query(description='是否启用（0停用 1启用）')] = None,
) -> ResponseSchemaModel[PageData[GetText2SqlTableDetail]]:
    page_data = await text2sql_service.get_selected_list(
        db=db,
        dataset_id=dataset_id,
        schema_name=schema_name,
        table_name=table_name,
        enabled=enabled,
    )
    return response_base.success(data=page_data)


@router.post(
    '/selected-tables',
    summary='挑选表',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:table:add')),
        DependsRBAC,
    ],
)
async def select_table(db: CurrentSessionTransaction, obj: CreateText2SqlTableParam) -> ResponseModel:
    await text2sql_service.select_table(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/selected-tables/{pk}',
    summary='更新已选表',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:table:edit')),
        DependsRBAC,
    ],
)
async def update_selected_table(
    db: CurrentSessionTransaction,
    pk: Annotated[int, Path(description='已选表 ID')],
    obj: UpdateText2SqlTableParam,
) -> ResponseModel:
    count = await text2sql_service.update_selected(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/selected-tables/{pk}',
    summary='取消挑选',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:table:del')),
        DependsRBAC,
    ],
)
async def unselect_table(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='已选表 ID')]
) -> ResponseModel:
    count = await text2sql_service.unselect_table(db=db, pk=pk)
    if count > 0:
        return response_base.success()
    return response_base.fail()


# ---------------- Few-shot 样例 ----------------


@router.get('/examples/all', summary='获取全部启用样例', dependencies=[DependsJwtAuth])
async def get_all_examples(
    db: CurrentSession,
    dataset_id: Annotated[int | None, Query(description='所属数据集 ID')] = None,
) -> ResponseSchemaModel[list[GetText2SqlExampleDetail]]:
    data = await text2sql_service.get_all_examples(db=db, dataset_id=dataset_id)
    return response_base.success(data=data)


@router.get('/examples/{pk}', summary='获取样例详情', dependencies=[DependsJwtAuth])
async def get_example(
    db: CurrentSession, pk: Annotated[int, Path(description='样例 ID')]
) -> ResponseSchemaModel[GetText2SqlExampleDetail]:
    data = await text2sql_service.get_example(db=db, pk=pk)
    return response_base.success(data=data)


@router.get(
    '/examples',
    summary='分页获取样例',
    dependencies=[
        DependsJwtAuth,
        DependsPagination,
    ],
)
async def get_examples(
    db: CurrentSession,
    dataset_id: Annotated[int | None, Query(description='所属数据集 ID')] = None,
    question: Annotated[str | None, Query(description='自然语言问题（模糊）')] = None,
    enabled: Annotated[int | None, Query(description='是否启用（0停用 1启用）')] = None,
) -> ResponseSchemaModel[PageData[GetText2SqlExampleDetail]]:
    page_data = await text2sql_service.get_example_list(
        db=db,
        dataset_id=dataset_id,
        question=question,
        enabled=enabled,
    )
    return response_base.success(data=page_data)


@router.post(
    '/examples',
    summary='新增样例',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:example:add')),
        DependsRBAC,
    ],
)
async def create_example(db: CurrentSessionTransaction, obj: CreateText2SqlExampleParam) -> ResponseModel:
    await text2sql_service.create_example(db=db, obj=obj)
    return response_base.success()


@router.put(
    '/examples/{pk}',
    summary='更新样例',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:example:edit')),
        DependsRBAC,
    ],
)
async def update_example(
    db: CurrentSessionTransaction,
    pk: Annotated[int, Path(description='样例 ID')],
    obj: UpdateText2SqlExampleParam,
) -> ResponseModel:
    count = await text2sql_service.update_example(db=db, pk=pk, obj=obj)
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/examples/{pk}',
    summary='删除样例',
    dependencies=[
        Depends(RequestPermission('ai:text2sql:example:del')),
        DependsRBAC,
    ],
)
async def delete_example(
    db: CurrentSessionTransaction, pk: Annotated[int, Path(description='样例 ID')]
) -> ResponseModel:
    count = await text2sql_service.delete_example(db=db, pk=pk)
    if count > 0:
        return response_base.success()
    return response_base.fail()
