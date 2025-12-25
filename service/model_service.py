from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.model import AIModel
from backend.plugin.ai.schema.model import CreateAIModelParam, DeleteAIModelParam, UpdateAIModelParam


class AIModelService:
    """AI 模型服务"""

    @staticmethod
    async def get(*, db: AsyncSession, pk: int) -> AIModel | None:
        """
        获取 AI 模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :return:
        """
        await ai_model_dao.get(db, pk)

    @staticmethod
    async def get_list(db: AsyncSession) -> dict[str, Any]:
        """
        获取 AI 模型列表

        :param db: 数据库会话
        :return:
        """
        ai_model_select = await ai_model_dao.get_select()
        return await paging_data(db, ai_model_select)

    @staticmethod
    async def get_all(*, db: AsyncSession) -> Sequence[AIModel]:
        """
        获取所有 AI 模型

        :param db: 数据库会话
        :return:
        """
        ai_providers = await ai_model_dao.get_all(db)
        return ai_providers

    @staticmethod
    async def create(*, db: AsyncSession, obj: CreateAIModelParam) -> None:
        """
        创建 AI 模型

        :param db: 数据库会话
        :param obj: 创建模型参数
        :return:
        """
        await ai_model_dao.create(db, obj)

    @staticmethod
    async def update(*, db: AsyncSession, pk: int, obj: UpdateAIModelParam) -> int:
        """
        更新 AI 模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :param obj: 更新模型参数
        :return:
        """
        count = await ai_model_dao.update(db, pk, obj)
        return count

    @staticmethod
    async def delete(*, db: AsyncSession, obj: DeleteAIModelParam) -> int:
        """
        删除 AI 模型

        :param db: 数据库会话
        :param obj: 模型 ID 列表
        :return:
        """
        count = await ai_model_dao.delete(db, obj.pks)
        return count


ai_model_service: AIModelService = AIModelService()
