from fastapi import APIRouter

from backend.common.response.response_schema import ResponseSchemaModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSession
from backend.plugin.ai.schema.model_option import GetAIModelOptionsDetail
from backend.plugin.ai.service.model_option_service import ai_model_option_service

router = APIRouter()


@router.get('', summary='获取模型选项', dependencies=[DependsJwtAuth])
async def get_ai_model_options(db: CurrentSession) -> ResponseSchemaModel[GetAIModelOptionsDetail]:
    data = await ai_model_option_service.get_all(db=db)
    return response_base.success(data=data)
