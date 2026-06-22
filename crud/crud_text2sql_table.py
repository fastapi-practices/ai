from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIText2SqlTable
from backend.plugin.ai.schema.text2sql import CreateText2SqlTableParam, UpdateText2SqlTableParam
from backend.utils.timezone import timezone


class CRUDText2SqlTable(CRUDPlus[AIText2SqlTable]):
    """Text2SQL 已选数据表数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIText2SqlTable | None:
        """
        获取已选表

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_by_name(
        self,
        db: AsyncSession,
        schema_name: str,
        table_name: str,
        dataset_id: int,
    ) -> AIText2SqlTable | None:
        """
        通过数据集+库名+表名获取已选表（同一数据集内表名唯一）

        :param db: 数据库会话
        :param schema_name: 库名/schema
        :param table_name: 表名
        :param dataset_id: 所属数据集 ID
        :return:
        """
        return await self.select_model_by_column(
            db,
            schema_name=schema_name,
            table_name=table_name,
            dataset_id=dataset_id,
            deleted=0,
        )

    async def get_enabled(self, db: AsyncSession, dataset_id: int | None = None) -> Sequence[AIText2SqlTable]:
        """
        获取全部启用的已选表（供 Agent 作为可见表集合）

        :param db: 数据库会话
        :param dataset_id: 数据集 ID；传入则仅返回该数据集的表，不传则返回全部
        :return:
        """
        filters: dict = {'enabled': 1, 'deleted': 0}
        if dataset_id is not None:
            filters['dataset_id'] = dataset_id
        return await self.select_models(db, **filters)

    async def get_all(self, db: AsyncSession) -> Sequence[AIText2SqlTable]:
        """
        获取全部已选表

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db, deleted=0)

    async def get_select(
        self,
        schema_name: str | None,
        table_name: str | None,
        enabled: int | None,
        dataset_id: int | None = None,
    ) -> Select:
        """
        获取已选表分页查询

        :param schema_name: 库名/schema
        :param table_name: 表名
        :param enabled: 是否启用
        :param dataset_id: 所属数据集 ID
        :return:
        """
        filters = {'deleted': 0}
        if dataset_id is not None:
            filters.update(dataset_id=dataset_id)
        if schema_name is not None:
            filters.update(schema_name__like=f'%{schema_name}%')
        if table_name is not None:
            filters.update(table_name__like=f'%{table_name}%')
        if enabled is not None:
            filters.update(enabled=enabled)
        return await self.select_order('sort', 'asc', **filters)

    async def create(self, db: AsyncSession, obj: CreateText2SqlTableParam) -> None:
        """
        创建已选表

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateText2SqlTableParam) -> int:
        """
        更新已选表

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete(self, db: AsyncSession, pk: int) -> int:
        """
        删除已选表（逻辑删除）

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        return await self.delete_model_by_column(
            db,
            logical_deletion=True,
            deleted_flag_column='deleted',
            deleted_flag_value=self.model.id,
            deleted_at_column='deleted_time',
            deleted_at_factory=timezone.now(),
            id=pk,
            deleted=0,
        )


text2sql_table_dao: CRUDText2SqlTable = CRUDText2SqlTable(AIText2SqlTable)
