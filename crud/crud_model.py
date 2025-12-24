from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIModel
from backend.plugin.ai.schema.model import CreateAIModelParam, UpdateAIModelParam


class CRUDAIModel(CRUDPlus[AIModel]):
    async def get(self, db: AsyncSession, pk: int) -> AIModel | None:
        """
        获取模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :return:
        """
        return await self.select_model(db, pk)

    async def get_by_model_and_provider(self, db: AsyncSession, model_id: str, provider_id: int) -> AIModel | None:
        """
        通过模型和供应商获取模型

        :param db: 数据库会话
        :param model_id: 模型
        :param provider_id: 供应商
        :return:
        """
        return await self.select_model_by_column(db, model_id=model_id, provider_id=provider_id)

    async def get_select(self) -> Select:
        """获取模型列表查询表达式"""
        return await self.select_order('id', 'desc')

    async def get_all(self, db: AsyncSession) -> Sequence[AIModel]:
        """
        获取所有模型

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db)

    async def create(self, db: AsyncSession, obj: CreateAIModelParam) -> None:
        """
        创建模型

        :param db: 数据库会话
        :param obj: 创建模型参数
        :return:
        """
        await self.create_model(db, obj)

    async def bulk_create(self, db: AsyncSession, objs: list[dict[str, Any]]) -> None:
        """
        批量创建模型

        :param db:数据库会话
        :param objs: 批量创建模型参数
        :return:
        """
        await self.bulk_create_models(db, objs)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateAIModelParam) -> int:
        """
        更新模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :param obj: 更新 模型参数
        :return:
        """
        return await self.update_model(db, pk, obj)

    async def delete(self, db: AsyncSession, pks: list[int]) -> int:
        """
        批量删除模型

        :param db: 数据库会话
        :param pks: 模型 ID 列表
        :return:
        """
        return await self.delete_model_by_column(db, allow_multiple=True, id__in=pks)

    async def delete_by_provider(self, db: AsyncSession, provider_id: int) -> int:
        """
        通过供应商 ID 删除模型

        :param db: 数据库会话
        :param provider_id: 供应商 ID
        :return:
        """
        return await self.delete_model_by_column(db, allow_multiple=True, provider_id=provider_id)


ai_model_dao: CRUDAIModel = CRUDAIModel(AIModel)
