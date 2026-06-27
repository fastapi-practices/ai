from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIText2SqlDataset
from backend.plugin.ai.schema.text2sql import CreateText2SqlDatasetParam, UpdateText2SqlDatasetParam
from backend.utils.timezone import timezone


class CRUDText2SqlDataset(CRUDPlus[AIText2SqlDataset]):
    """Text2SQL 数据集数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIText2SqlDataset | None:
        """
        获取数据集

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_all(self, db: AsyncSession) -> Sequence[AIText2SqlDataset]:
        """
        获取全部数据集

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db, deleted=0)

    async def get_enabled(self, db: AsyncSession) -> Sequence[AIText2SqlDataset]:
        """
        获取全部启用的数据集（供 chat 选择器）

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db, enabled=1, deleted=0)

    async def get_by_name(self, db: AsyncSession, name: str) -> AIText2SqlDataset | None:
        """
        通过名称获取数据集

        :param db: 数据库会话
        :param name: 数据集名称
        :return:
        """
        return await self.select_model_by_column(db, name=name, deleted=0)

    async def get_select(self, name: str | None, enabled: int | None) -> Select:
        """
        获取数据集分页查询

        :param name: 数据集名称（模糊）
        :param enabled: 是否启用
        :return:
        """
        filters = {'deleted': 0}
        if name is not None:
            filters.update(name__like=f'%{name}%')
        if enabled is not None:
            filters.update(enabled=enabled)
        return await self.select_order('sort', 'asc', **filters)

    async def create(self, db: AsyncSession, obj: CreateText2SqlDatasetParam) -> None:
        """
        创建数据集

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateText2SqlDatasetParam) -> int:
        """
        更新数据集

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete(self, db: AsyncSession, pk: int) -> int:
        """
        删除数据集（逻辑删除）

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


text2sql_dataset_dao: CRUDText2SqlDataset = CRUDText2SqlDataset(AIText2SqlDataset)
