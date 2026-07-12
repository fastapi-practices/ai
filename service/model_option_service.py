from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import StatusType
from backend.plugin.ai.crud.crud_default_model import ai_default_model_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIDefaultModelScene
from backend.plugin.ai.schema.default_model import GetAIDefaultModelDetail
from backend.plugin.ai.schema.model import GetAIModelDetail
from backend.plugin.ai.schema.model_option import GetAIModelOptionsDetail, GetAIProviderModelOptionDetail

if TYPE_CHECKING:
    from backend.plugin.ai.model import AIModel


class AIModelOptionService:
    """AI 模型选项服务"""

    @staticmethod
    async def get_all(*, db: AsyncSession) -> GetAIModelOptionsDetail:
        """
        获取模型选项

        :param db: 数据库会话
        :return:
        """
        providers = await ai_provider_dao.get_all(db)
        provider_by_id = {provider.id: provider for provider in providers}
        default_model = await ai_default_model_dao.get_by_scene(db, AIDefaultModelScene.assistant)
        models = await ai_model_dao.get_all_by_providers(
            db,
            [provider.id for provider in providers],
            status=StatusType.enable.value,
        )
        models_by_provider: dict[int, list[GetAIModelDetail]] = {provider.id: [] for provider in providers}
        model_by_pair: dict[tuple[int, str], AIModel] = {}
        for model in models:
            model_by_pair[model.provider_id, model.model_id] = model
            if model.status == StatusType.enable:
                models_by_provider[model.provider_id].append(GetAIModelDetail.model_validate(model))

        provider_options = [
            GetAIProviderModelOptionDetail(
                id=provider.id,
                name=provider.name,
                type=provider.type,
                status=provider.status,
                models=models_by_provider[provider.id],
            )
            for provider in providers
        ]
        default_model_detail = None
        if default_model and default_model.status == StatusType.enable:
            default_provider = provider_by_id.get(default_model.provider_id)
            default_ai_model = model_by_pair.get((default_model.provider_id, default_model.model_id))
            if default_provider and default_ai_model and default_ai_model.status == StatusType.enable:
                default_model_detail = GetAIDefaultModelDetail(
                    id=default_model.id,
                    scene=AIDefaultModelScene(default_model.scene),
                    provider_id=default_model.provider_id,
                    provider_name=default_provider.name,
                    provider_type=default_provider.type,
                    model_id=default_model.model_id,
                    status=default_model.status,
                    created_time=default_model.created_time,
                    updated_time=default_model.updated_time,
                )
        return GetAIModelOptionsDetail(
            providers=provider_options,
            default_model=default_model_detail,
        )


ai_model_option_service: AIModelOptionService = AIModelOptionService()
