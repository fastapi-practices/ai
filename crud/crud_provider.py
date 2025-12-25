from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIProvider
from backend.plugin.ai.schema.provider import CreateAIProviderParam, UpdateAIProviderParam


class CRUDAIProvider(CRUDPlus[AIProvider]):
    async def get(self, db: AsyncSession, pk: int) -> AIProvider | None:
        """
        获取供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        return await self.select_model(db, pk)

    async def get_select(self) -> Select:
        """获取供应商列表查询表达式"""
        return await self.select_order('id', 'desc')

    async def get_all(self, db: AsyncSession) -> Sequence[AIProvider]:
        """
        获取所有供应商

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db)

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
        :param obj: 更新 供应商参数
        :return:
        """
        return await self.update_model(db, pk, obj)

    async def delete(self, db: AsyncSession, pks: list[int]) -> int:
        """
        批量删除供应商

        :param db: 数据库会话
        :param pks: 供应商 ID 列表
        :return:
        """
        return await self.delete_model_by_column(db, allow_multiple=True, id__in=pks)


ai_provider_dao: CRUDAIProvider = CRUDAIProvider(AIProvider)
