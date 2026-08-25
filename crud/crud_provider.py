from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.common.enums import StatusType
from backend.plugin.ai.model import AIProvider
from backend.plugin.ai.schema.provider import CreateAIProviderParam, UpdateAIProviderParam
from backend.utils.timezone import timezone


class CRUDAIProvider(CRUDPlus[AIProvider]):
    """AI 供应商数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIProvider | None:
        """
        获取供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_by_ids(self, db: AsyncSession, pks: list[int]) -> Sequence[AIProvider]:
        """
        批量获取供应商

        :param db: 数据库会话
        :param pks: 供应商 ID 列表
        :return:
        """
        return await self.select_models(db, id__in=pks, deleted=0)

    async def get_select(self, name: str | None, type: int | None, status: int | None) -> Select:
        """
        获取供应商列表查询表达式

        :param name: 供应商名称
        :param type: 供应商类型
        :param status: 状态
        :return:
        """
        filters = {'deleted': 0}

        if name is not None:
            filters['name__like'] = f'%{name}%'
        if type is not None:
            filters['type'] = type
        if status is not None:
            filters['status'] = status

        return await self.select_order('id', 'desc', **filters)

    async def get_all(self, db: AsyncSession) -> Sequence[AIProvider]:
        """
        获取所有供应商

        :param db: 数据库会话
        :return:
        """
        return await self.select_models_order(db, 'id', 'desc', status=StatusType.enable.value, deleted=0)

    async def create(self, db: AsyncSession, obj: CreateAIProviderParam) -> None:
        """
        创建供应商

        :param db: 数据库会话
        :param obj: 创建供应商参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateAIProviderParam) -> int:
        """
        更新供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :param obj: 更新供应商参数
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete(self, db: AsyncSession, pks: list[int]) -> int:
        """
        批量删除供应商

        :param db: 数据库会话
        :param pks: 供应商 ID 列表
        :return:
        """
        return await self.delete_model_by_column(
            db,
            allow_multiple=True,
            logical_deletion=True,
            deleted_flag_column='deleted',
            deleted_flag_value=self.model.id,
            deleted_at_column='deleted_time',
            deleted_at_factory=timezone.now(),
            id__in=pks,
            deleted=0,
        )


ai_provider_dao: CRUDAIProvider = CRUDAIProvider(AIProvider)
