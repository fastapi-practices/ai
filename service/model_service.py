from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import StatusType
from backend.common.exception import errors
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.model import AIModel
from backend.plugin.ai.schema.model import CreateAIModelParam, DeleteAIModelParam, UpdateAIModelParam


class AIModelService:
    """AI 模型服务"""

    @staticmethod
    async def get(*, db: AsyncSession, pk: int) -> AIModel:
        """
        获取 AI 模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :return:
        """
        ai_model = await ai_model_dao.get(db, pk)
        if not ai_model:
            raise errors.NotFoundError(msg='模型不存在')
        return ai_model

    @staticmethod
    async def get_list(
        *,
        db: AsyncSession,
        provider_id: int | None,
        model_id: str | None,
        status: int | None,
    ) -> dict[str, Any]:
        """
        获取 AI 模型列表

        :param db: 数据库会话
        :param provider_id: 供应商 ID
        :param model_id: 模型 ID
        :param status: 状态
        :return:
        """
        ai_model_select = await ai_model_dao.get_select(provider_id, model_id, status)
        return await paging_data(db, ai_model_select)

    @staticmethod
    async def get_all(*, db: AsyncSession, provider_id: int) -> Sequence[AIModel]:
        """
        获取所有 AI 模型

        :param db: 数据库会话
        :param provider_id: 供应商 ID
        :return:
        """
        ai_models = await ai_model_dao.get_all(db, provider_id=provider_id, status=StatusType.enable)
        return ai_models

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
        ai_model = await ai_model_dao.get(db, pk)
        if not ai_model:
            raise errors.NotFoundError(msg='模型不存在')
        return await ai_model_dao.update(db, pk, obj)

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
