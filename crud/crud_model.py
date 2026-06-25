from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIModel
from backend.plugin.ai.schema.model import CreateAIModelParam, UpdateAIModelParam
from backend.utils.timezone import timezone


class CRUDAIModel(CRUDPlus[AIModel]):
    """AI 模型数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIModel | None:
        """
        获取模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_by_model_and_provider(self, db: AsyncSession, model_id: str, provider_id: int) -> AIModel | None:
        """
        通过模型和供应商获取模型

        :param db: 数据库会话
        :param model_id: 模型
        :param provider_id: 供应商
        :return:
        """
        return await self.select_model_by_column(db, model_id=model_id, provider_id=provider_id, deleted=0)

    async def get_select(self, provider_id: int | None, model_id: str | None, status: int | None) -> Select:
        """
        获取模型列表查询表达式

        :param provider_id: 供应商 ID
        :param model_id: 模型 ID
        :param status: 状态
        :return:
        """
        filters = {'deleted': 0}

        if provider_id is not None:
            filters['provider_id'] = provider_id
        if model_id is not None:
            filters['model_id__like'] = f'%{model_id}%'
        if status is not None:
            filters['status'] = status

        return await self.select_order('id', 'desc', **filters)

    async def get_all(self, db: AsyncSession, provider_id: int, *, status: int | None = None) -> Sequence[AIModel]:
        """
        获取所有模型

        :param db: 数据库会话
        :param provider_id: 供应商 ID
        :param status: 状态
        :return:
        """
        filters = {'provider_id': provider_id, 'deleted': 0}

        if status is not None:
            filters['status'] = status

        return await self.select_models(db, **filters)

    async def get_by_ids(self, db: AsyncSession, pks: list[int]) -> Sequence[AIModel]:
        """
        批量获取模型

        :param db: 数据库会话
        :param pks: 模型 ID 列表
        :return:
        """
        if not pks:
            return []
        return await self.select_models(db, id__in=pks, deleted=0)

    async def get_by_provider_model_pairs(self, db: AsyncSession, pairs: list[tuple[int, str]]) -> Sequence[AIModel]:
        """
        批量获取已存在的模型

        :param db: 数据库会话
        :param pairs: 供应商 ID 与模型 ID 组合列表
        :return:
        """
        if not pairs:
            return []

        provider_ids = list({provider_id for provider_id, _ in pairs})
        model_ids = list({model_id for _, model_id in pairs})
        pair_set = set(pairs)
        models = await self.select_models(db, provider_id__in=provider_ids, model_id__in=model_ids, deleted=0)
        return [model for model in models if (model.provider_id, model.model_id) in pair_set]

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
        now = timezone.now()
        payloads = [
            {
                **obj,
                'created_time': obj.get('created_time', now),
            }
            for obj in objs
        ]
        await self.bulk_create_models(db, payloads)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateAIModelParam) -> int:
        """
        更新模型

        :param db: 数据库会话
        :param pk: 模型 ID
        :param obj: 更新 模型参数
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete(self, db: AsyncSession, pks: list[int]) -> int:
        """
        批量删除模型

        :param db: 数据库会话
        :param pks: 模型 ID 列表
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

    async def delete_by_provider(self, db: AsyncSession, provider_id: int) -> int:
        """
        通过供应商 ID 删除模型

        :param db: 数据库会话
        :param provider_id: 供应商 ID
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
            provider_id=provider_id,
            deleted=0,
        )

    async def delete_by_providers(self, db: AsyncSession, provider_ids: list[int]) -> int:
        """
        通过供应商 ID 列表批量删除模型

        :param db: 数据库会话
        :param provider_ids: 供应商 ID 列表
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
            provider_id__in=provider_ids,
            deleted=0,
        )


ai_model_dao: CRUDAIModel = CRUDAIModel(AIModel)
