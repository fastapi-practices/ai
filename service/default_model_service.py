from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import StatusType
from backend.common.exception import errors
from backend.plugin.ai.crud.crud_default_model import ai_default_model_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIDefaultModelScene
from backend.plugin.ai.schema.default_model import GetAIDefaultModelDetail, UpdateAIDefaultModelParam


class AIDefaultModelService:
    """AI 默认模型服务类"""

    @staticmethod
    async def get(*, db: AsyncSession, scene: AIDefaultModelScene) -> GetAIDefaultModelDetail:
        """
        获取默认模型

        :param db: 数据库会话
        :param scene: 默认模型场景
        :return:
        """
        default_model = await ai_default_model_dao.get_by_scene(db, scene)
        if not default_model:
            raise errors.NotFoundError(msg='默认模型配置不存在')
        if default_model.status != StatusType.enable:
            raise errors.RequestError(msg='默认模型配置已停用')
        provider = await ai_provider_dao.get(db, default_model.provider_id)
        if not provider:
            raise errors.NotFoundError(msg='默认模型供应商不存在')
        if provider.status != StatusType.enable:
            raise errors.RequestError(msg='默认模型供应商已停用')
        model = await ai_model_dao.get_by_model_and_provider(db, default_model.model_id, default_model.provider_id)
        if not model:
            raise errors.NotFoundError(msg='默认模型不存在')
        if model.status != StatusType.enable:
            raise errors.RequestError(msg='默认模型已停用')
        return GetAIDefaultModelDetail(
            id=default_model.id,
            scene=AIDefaultModelScene(default_model.scene),
            provider_id=default_model.provider_id,
            provider_name=provider.name,
            provider_type=provider.type,
            model_id=default_model.model_id,
            status=default_model.status,
            created_time=default_model.created_time,
            updated_time=default_model.updated_time,
        )

    @staticmethod
    async def update(*, db: AsyncSession, scene: AIDefaultModelScene, obj: UpdateAIDefaultModelParam) -> None:
        """
        更新默认模型

        :param db: 数据库会话
        :param scene: 默认模型场景
        :param obj: 默认模型配置参数
        :return:
        """
        provider = await ai_provider_dao.get(db, obj.provider_id)
        if not provider:
            raise errors.NotFoundError(msg='供应商不存在')
        if provider.status != StatusType.enable:
            raise errors.RequestError(msg='供应商已停用')
        model = await ai_model_dao.get_by_model_and_provider(db, obj.model_id, obj.provider_id)
        if not model:
            raise errors.NotFoundError(msg='模型不存在')
        if model.status != StatusType.enable:
            raise errors.RequestError(msg='模型已停用')
        default_model = await ai_default_model_dao.get_by_scene(db, scene)
        if default_model:
            await ai_default_model_dao.update(db, default_model.id, obj)
            return
        await ai_default_model_dao.create(db, scene, obj)


ai_default_model_service: AIDefaultModelService = AIDefaultModelService()
