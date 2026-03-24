from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import StatusType
from backend.common.exception import errors
from backend.common.log import log
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.model import AIProvider
from backend.plugin.ai.schema.model import CreateAIModelParam
from backend.plugin.ai.schema.provider import (
    CreateAIProviderParam,
    DeleteAIProviderParam,
    GetAIProviderModelDetail,
    UpdateAIProviderParam,
)
from backend.plugin.ai.utils.provider_url import normalize_provider_api_host
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
        if ai_provider.status != StatusType.enable:
            raise errors.RequestError(msg='当前供应商已停用，无法获取模型列表')
        if ai_provider.type not in {AIProviderType.openai, AIProviderType.xai, AIProviderType.openrouter}:
            raise errors.RequestError(msg='当前供应商暂不支持自动同步模型，请手动维护模型列表')
        url = f'{normalize_provider_api_host(ai_provider.type, ai_provider.api_host)}/models'
        headers = {'Authorization': f'Bearer {ai_provider.api_key}'}
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
                return [GetAIProviderModelDetail(**data) for data in payload['data']]
            except httpx.HTTPError as e:
                log.error(f'获取供应商模型列表失败：{e}')
                raise errors.ForbiddenError(msg='获取供应商模型列表失败，请稍后重试')
            except ValueError as e:
                log.error(f'供应商模型列表 JSON 解析失败：{e}')
                raise errors.RequestError(msg='供应商返回的模型数据不是合法 JSON') from e
            except (KeyError, TypeError, ValidationError) as e:
                log.error(f'供应商模型列表数据格式错误：{e}')
                raise errors.RequestError(msg='供应商返回的模型数据格式不正确') from e

    async def sync_models(self, *, db: AsyncSession, pk: int) -> None:
        """
        同步供应商模型

        :param db: 数据库会话
        :param pk: 供应商 ID
        :return:
        """
        existing_models = await ai_model_dao.get_all(db, provider_id=pk)
        existing_status = {model.model_id: model.status for model in existing_models}
        provider_models = await self.get_models(db=db, pk=pk)
        await ai_model_dao.delete_by_provider(db, pk)
        if not provider_models:
            return

        await ai_model_dao.bulk_create(
            db,
            [
                {
                    **CreateAIModelParam(
                        provider_id=pk,
                        model_id=obj.id,
                        status=existing_status.get(obj.id, StatusType.enable),  # type: ignore
                    ).model_dump(),
                    'created_time': timezone.now(),
                }
                for obj in provider_models
            ],
        )

    @staticmethod
    async def get_list(
        *,
        db: AsyncSession,
        name: str | None,
        type: int | None,
        status: int | None,
    ) -> dict[str, Any]:
        """
        获取 AI 供应商列表

        :param db: 数据库会话
        :param name: 供应商名称
        :param type: 供应商类型
        :param status: 状态
        :return:
        """
        ai_provider_select = await ai_provider_dao.get_select(name, type, status)
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
        parsed = urlsplit(obj.api_host.strip())
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise errors.RequestError(msg='API Host 必须是合法的 HTTP(S) 地址')
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
        ai_provider = await ai_provider_dao.get(db, pk)
        if not ai_provider:
            raise errors.NotFoundError(msg='供应商不存在')
        parsed = urlsplit(obj.api_host.strip())
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise errors.RequestError(msg='API Host 必须是合法的 HTTP(S) 地址')
        return await ai_provider_dao.update(db, pk, obj)

    @staticmethod
    async def delete(*, db: AsyncSession, obj: DeleteAIProviderParam) -> int:
        """
        删除 AI 供应商

        :param db: 数据库会话
        :param obj: 供应商 ID 列表
        :return:
        """
        await ai_model_dao.delete_by_providers(db, obj.pks)
        count = await ai_provider_dao.delete(db, obj.pks)
        return count


ai_provider_service: AIProviderService = AIProviderService()
