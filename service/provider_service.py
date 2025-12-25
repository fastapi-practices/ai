from collections.abc import Sequence
from typing import Any

import httpx

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import StatusType
from backend.common.exception import errors
from backend.common.log import log
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.model import AIProvider
from backend.plugin.ai.schema.model import CreateAIModelParam
from backend.plugin.ai.schema.provider import (
    CreateAIProviderParam,
    DeleteAIProviderParam,
    GetAIProviderModelDetail,
    UpdateAIProviderParam,
)
from backend.utils.timezone import timezone


class AIProviderService:
    @staticmethod
    async def get(*, db: AsyncSession, pk: int) -> AIProvider:
        """
        获取 AI 供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        ai_provider = await ai_provider_dao.get(db, pk)
        if not ai_provider:
            raise errors.NotFoundError(msg='供应商不存在')
        return ai_provider

    async def get_models(self, *, db: AsyncSession, pk: int) -> list[GetAIProviderModelDetail]:
        """获取供应商模型"""
        ai_provider = await self.get(db=db, pk=pk)
        async with httpx.AsyncClient(timeout=10) as client:
            url = f'{ai_provider.api_host}/v1/models'
            headers = {'Authorization': f'Bearer {ai_provider.api_key}'}
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
            except Exception as e:
                log.error(f'获取供应商模型列表失败：{e}')
                raise errors.ForbiddenError(msg='获取供应商模型列表失败，请稍后重试')
            else:
                return [GetAIProviderModelDetail(**data) for data in response.json()['data']]

    async def sync_models(self, *, db: AsyncSession, pk: int) -> None:
        """
        同步供应商模型

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        provider_models = await self.get_models(db=db, pk=pk)
        await ai_model_dao.delete_by_provider(db, pk)
        await ai_model_dao.bulk_create(
            db,
            [
                {
                    **CreateAIModelParam(
                        provider_id=pk,
                        model_id=obj.id,
                        owned_by=obj.owned_by,
                        status=StatusType.enable,
                    ).model_dump(),
                    'created_time': timezone.now(),
                }
                for obj in provider_models
            ],
        )

    @staticmethod
    async def get_list(db: AsyncSession) -> dict[str, Any]:
        """
        获取 AI 供应商列表

        :param db: 数据库会话
        :return:
        """
        ai_provider_select = await ai_provider_dao.get_select()
        return await paging_data(db, ai_provider_select)

    @staticmethod
    async def get_all(*, db: AsyncSession) -> Sequence[AIProvider]:
        """
        获取所有 AI 供应商

        :param db: 数据库会话
        :return:
        """
        ai_providers = await ai_provider_dao.get_all(db)
        return ai_providers

    @staticmethod
    async def create(*, db: AsyncSession, obj: CreateAIProviderParam) -> None:
        """
        创建 AI 供应商

        :param db: 数据库会话
        :param obj: 创建供应商参数
        :return:
        """
        if obj.api_host.endswith('/'):
            raise errors.RequestError(msg='API 请求地址不能以 `/` 结尾')
        await ai_provider_dao.create(db, obj)

    @staticmethod
    async def update(*, db: AsyncSession, pk: int, obj: UpdateAIProviderParam) -> int:
        """
        更新 AI 供应商

        :param db: 数据库会话
        :param pk: 供应商 ID
        :param obj: 更新供应商参数
        :return:
        """
        count = await ai_provider_dao.update(db, pk, obj)
        return count

    @staticmethod
    async def delete(*, db: AsyncSession, obj: DeleteAIProviderParam) -> int:
        """
        删除 AI 供应商

        :param db: 数据库会话
        :param obj: 供应商 ID 列表
        :return:
        """
        count = await ai_provider_dao.delete(db, obj.pks)
        return count


ai_provider_service: AIProviderService = AIProviderService()
