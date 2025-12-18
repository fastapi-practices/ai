from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.model import AiProvider
from backend.plugin.ai.schema.provider import CreateAiProviderParam, DeleteAiProviderParam, UpdateAiProviderParam


class AiProviderService:
    @staticmethod
    async def get(*, db: AsyncSession, pk: int) -> AiProvider:
        """
        获取供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        ai_provider = await ai_provider_dao.get(db, pk)
        if not ai_provider:
            raise errors.NotFoundError(msg='供应商不存在')
        return ai_provider

    @staticmethod
    async def get_list(db: AsyncSession) -> dict[str, Any]:
        """
        获取供应商列表

        :param db: 数据库会话
        :return:
        """
        ai_provider_select = await ai_provider_dao.get_select()
        return await paging_data(db, ai_provider_select)

    @staticmethod
    async def get_all(*, db: AsyncSession) -> Sequence[AiProvider]:
        """
        获取所有供应商

        :param db: 数据库会话
        :return:
        """
        ai_providers = await ai_provider_dao.get_all(db)
        return ai_providers

    @staticmethod
    async def create(*, db: AsyncSession, obj: CreateAiProviderParam) -> None:
        """
        创建供应商

        :param db: 数据库会话
        :param obj: 创建供应商参数
        :return:
        """
        await ai_provider_dao.create(db, obj)

    @staticmethod
    async def update(*, db: AsyncSession, pk: int, obj: UpdateAiProviderParam) -> int:
        """
        更新供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :param obj: 更新供应商参数
        :return:
        """
        count = await ai_provider_dao.update(db, pk, obj)
        return count

    @staticmethod
    async def delete(*, db: AsyncSession, obj: DeleteAiProviderParam) -> int:
        """
        删除供应商

        :param db: 数据库会话
        :param obj: 供应商 ID 列表
        :return:
        """
        count = await ai_provider_dao.delete(db, obj.pks)
        return count


ai_provider_service: AiProviderService = AiProviderService()
