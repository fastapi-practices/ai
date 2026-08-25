from fastapi import APIRouter, Depends

from backend.common.response.response_schema import ResponseModel, ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.common.security.permission import RequestPermission
from backend.common.security.rbac import DependsRBAC
from backend.database.db import CurrentSession, CurrentSessionTransaction
from backend.plugin.ai.enums import AIDefaultModelScene
from backend.plugin.ai.schema.default_model import GetAIDefaultModelDetail, UpdateAIDefaultModelParam
from backend.plugin.ai.service.default_model_service import ai_default_model_service

router = APIRouter()


@router.get('/assistant', summary='获取默认助手模型配置', dependencies=[DependsJwtAuth])
async def get_ai_assistant_default_model(db: CurrentSession) -> ResponseSchemaModel[GetAIDefaultModelDetail]:
    data = await ai_default_model_service.get(db=db, scene=AIDefaultModelScene.assistant)
    return response_base.success(data=data)


@router.put(
    '/assistant',
    summary='更新默认助手模型配置',
    dependencies=[
        Depends(RequestPermission('ai:default-model:edit')),
        DependsRBAC,
    ],
)
async def update_ai_assistant_default_model(
    db: CurrentSessionTransaction,
    obj: UpdateAIDefaultModelParam,
) -> ResponseModel:
    await ai_default_model_service.update(db=db, scene=AIDefaultModelScene.assistant, obj=obj)
    return response_base.success()
