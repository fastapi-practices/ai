from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import paging_data
from backend.core.conf import settings
from backend.plugin.ai.crud.crud_text2sql_dataset import text2sql_dataset_dao
from backend.plugin.ai.crud.crud_text2sql_example import text2sql_example_dao
from backend.plugin.ai.crud.crud_text2sql_table import text2sql_table_dao
from backend.plugin.ai.model import AIText2SqlDataset, AIText2SqlExample, AIText2SqlTable
from backend.plugin.ai.schema.text2sql import (
    CreateText2SqlDatasetParam,
    CreateText2SqlExampleParam,
    CreateText2SqlTableParam,
    Text2SqlDatasetEnabled,
    Text2SqlTableSelectable,
    UpdateText2SqlDatasetParam,
    UpdateText2SqlExampleParam,
    UpdateText2SqlTableParam,
)
from backend.plugin.ai.text2sql.schema_meta import get_columns, get_tables


class Text2SqlService:
    """Text2SQL 服务类"""

    # ---------------- 数据集 ----------------

    @staticmethod
    async def get_dataset(*, db: AsyncSession, pk: int) -> AIText2SqlDataset:
        """
        获取数据集详情

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_dataset_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='数据集不存在')
        return row

    @staticmethod
    async def get_all_datasets(*, db: AsyncSession) -> Sequence[AIText2SqlDataset]:
        """
        获取全部数据集

        :param db: 数据库会话
        :return:
        """
        return await text2sql_dataset_dao.get_all(db)

    @staticmethod
    async def get_enabled_datasets(*, db: AsyncSession) -> Sequence[Text2SqlDatasetEnabled]:
        """
        获取全部启用的数据集（供 chat 选择器）

        :param db: 数据库会话
        :return:
        """
        rows = await text2sql_dataset_dao.get_enabled(db)
        return [
            Text2SqlDatasetEnabled(id=row.id, name=row.name, description=row.description)
            for row in rows
        ]

    @staticmethod
    async def get_dataset_list(
        *,
        db: AsyncSession,
        name: str | None,
        enabled: int | None,
    ) -> dict[str, Any]:
        """
        分页获取数据集

        :param db: 数据库会话
        :param name: 数据集名称（模糊）
        :param enabled: 是否启用
        :return:
        """
        sel = await text2sql_dataset_dao.get_select(name=name, enabled=enabled)
        return await paging_data(db, sel)

    @staticmethod
    async def create_dataset(*, db: AsyncSession, obj: CreateText2SqlDatasetParam) -> None:
        """
        创建数据集

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        existing = await text2sql_dataset_dao.get_by_name(db, obj.name)
        if existing:
            raise errors.ForbiddenError(msg='数据集名称已存在')
        await text2sql_dataset_dao.create(db, obj)

    @staticmethod
    async def update_dataset(*, db: AsyncSession, pk: int, obj: UpdateText2SqlDatasetParam) -> int:
        """
        更新数据集

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        row = await text2sql_dataset_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='数据集不存在')
        return await text2sql_dataset_dao.update(db, pk, obj)

    @staticmethod
    async def delete_dataset(*, db: AsyncSession, pk: int) -> int:
        """
        删除数据集（逻辑删除；其下表与样例保留但不再被该数据集聚合）

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_dataset_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='数据集不存在')
        return await text2sql_dataset_dao.delete(db, pk)

    # ---------------- 数据源管理（已选表） ----------------

    @staticmethod
    async def list_selectable_tables(
        *,
        db: AsyncSession,
        dataset_id: int,
        table_schema: str | None = None,
    ) -> Sequence[Text2SqlTableSelectable]:
        """
        列出可挑选的数据库表（反查 information_schema），并标记在该数据集内是否已挑选

        :param db: 数据库会话
        :param dataset_id: 所属数据集 ID
        :param table_schema: 库名/schema，缺省取 AI_TEXT2SQL_SCHEMA
        :return:
        """
        schema = table_schema or settings.AI_TEXT2SQL_SCHEMA
        rows = await get_tables(db, schema)
        selected = await text2sql_table_dao.select_models(db, dataset_id=dataset_id, deleted=0)
        selected_names = {row.table_name for row in selected}
        return [
            Text2SqlTableSelectable(
                table_name=row['table_name'],
                table_comment=row['table_comment'],
                selected=row['table_name'] in selected_names,
            )
            for row in rows
        ]

    @staticmethod
    async def get_table_columns(
        *,
        db: AsyncSession,
        table_name: str,
        table_schema: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取表列信息（供列预览与拼装 DDL 上下文）

        :param db: 数据库会话
        :param table_name: 表名
        :param table_schema: 库名/schema
        :return:
        """
        schema = table_schema or settings.AI_TEXT2SQL_SCHEMA
        rows = await get_columns(db, schema, table_name)
        return [dict(row) for row in rows]

    @staticmethod
    async def get_selected(*, db: AsyncSession, pk: int) -> AIText2SqlTable:
        """
        获取已选表详情

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_table_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='已选表记录不存在')
        return row

    @staticmethod
    async def get_all_selected(*, db: AsyncSession) -> Sequence[AIText2SqlTable]:
        """
        获取全部已选表

        :param db: 数据库会话
        :return:
        """
        return await text2sql_table_dao.get_all(db)

    @staticmethod
    async def get_enabled(*, db: AsyncSession, dataset_id: int | None = None) -> Sequence[AIText2SqlTable]:
        """
        获取全部启用的已选表（Agent 可见表集合）

        :param db: 数据库会话
        :param dataset_id: 数据集 ID；传入则仅返回该数据集的表
        :return:
        """
        return await text2sql_table_dao.get_enabled(db, dataset_id=dataset_id)

    @staticmethod
    async def get_selected_list(
        *,
        db: AsyncSession,
        dataset_id: int | None,
        schema_name: str | None,
        table_name: str | None,
        enabled: int | None,
    ) -> dict[str, Any]:
        """
        分页获取已选表

        :param db: 数据库会话
        :param dataset_id: 所属数据集 ID
        :param schema_name: 库名/schema
        :param table_name: 表名
        :param enabled: 是否启用
        :return:
        """
        sel = await text2sql_table_dao.get_select(
            schema_name=schema_name,
            table_name=table_name,
            enabled=enabled,
            dataset_id=dataset_id,
        )
        return await paging_data(db, sel)

    @staticmethod
    async def select_table(*, db: AsyncSession, obj: CreateText2SqlTableParam) -> None:
        """
        挑选表（新增到已选表，同一数据集内表名唯一）

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        existing = await text2sql_table_dao.get_by_name(
            db,
            schema_name=obj.schema_name,
            table_name=obj.table_name,
            dataset_id=obj.dataset_id,
        )
        if existing:
            raise errors.ForbiddenError(msg='该数据集内此表已挑选')
        cols = await get_columns(db, obj.schema_name, obj.table_name)
        if not cols:
            raise errors.NotFoundError(msg='数据库表不存在')
        await text2sql_table_dao.create(db, obj)

    @staticmethod
    async def update_selected(*, db: AsyncSession, pk: int, obj: UpdateText2SqlTableParam) -> int:
        """
        更新已选表

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        row = await text2sql_table_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='已选表记录不存在')
        return await text2sql_table_dao.update(db, pk, obj)

    @staticmethod
    async def unselect_table(*, db: AsyncSession, pk: int) -> int:
        """
        取消挑选（逻辑删除已选表）

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_table_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='已选表记录不存在')
        return await text2sql_table_dao.delete(db, pk)

    # ---------------- Few-shot 样例 ----------------

    @staticmethod
    async def get_example(*, db: AsyncSession, pk: int) -> AIText2SqlExample:
        """
        获取样例详情

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_example_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='样例不存在')
        return row

    @staticmethod
    async def get_all_examples(*, db: AsyncSession, dataset_id: int | None = None) -> Sequence[AIText2SqlExample]:
        """
        获取全部样例

        :param db: 数据库会话
        :param dataset_id: 数据集 ID；传入则仅返回该数据集的样例
        :return:
        """
        return await text2sql_example_dao.get_all_enabled(db, dataset_id=dataset_id)

    @staticmethod
    async def get_example_list(
        *,
        db: AsyncSession,
        dataset_id: int | None,
        question: str | None,
        enabled: int | None,
    ) -> dict[str, Any]:
        """
        分页获取样例

        :param db: 数据库会话
        :param dataset_id: 所属数据集 ID
        :param question: 自然语言问题（模糊）
        :param enabled: 是否启用
        :return:
        """
        sel = await text2sql_example_dao.get_select(
            question=question,
            enabled=enabled,
            dataset_id=dataset_id,
        )
        return await paging_data(db, sel)

    @staticmethod
    async def create_example(*, db: AsyncSession, obj: CreateText2SqlExampleParam) -> None:
        """
        创建样例

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        await text2sql_example_dao.create(db, obj)

    @staticmethod
    async def update_example(*, db: AsyncSession, pk: int, obj: UpdateText2SqlExampleParam) -> int:
        """
        更新样例

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        row = await text2sql_example_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='样例不存在')
        return await text2sql_example_dao.update(db, pk, obj)

    @staticmethod
    async def delete_example(*, db: AsyncSession, pk: int) -> int:
        """
        删除样例

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        row = await text2sql_example_dao.get(db, pk)
        if not row:
            raise errors.NotFoundError(msg='样例不存在')
        return await text2sql_example_dao.delete(db, pk)

    @staticmethod
    async def get_examples_for(
        *,
        db: AsyncSession,
        tables: set[str],
        dataset_id: int | None = None,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """
        按命中表召回 Few-shot 样例（供 Agent 提升精度）

        无 related_tables 的样例视为通用样例，始终召回。

        :param db: 数据库会话
        :param tables: 本次涉及的表名集合
        :param dataset_id: 数据集 ID；传入则仅在该数据集样例中召回
        :param limit: 最多召回条数
        :return:
        """
        examples = await text2sql_example_dao.get_all_enabled(db, dataset_id=dataset_id)
        table_set = {name.lower() for name in tables}
        matched: list[AIText2SqlExample] = []
        for example in examples:
            related = {name.strip().lower() for name in (example.related_tables or '').split(',') if name.strip()}
            if not related or (related & table_set):
                matched.append(example)
        return [{'question': e.question, 'sql': e.sql} for e in matched[:limit]]


text2sql_service: Text2SqlService = Text2SqlService()
