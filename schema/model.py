from datetime import datetime

from pydantic import ConfigDict, Field

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase


class AIModelSchemaBase(SchemaBase):
    """AI 模型基础模型"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    owned_by: str = Field(description='拥有该模型的组织')
    status: StatusType = Field(description='状态')
    remark: str | None = Field(default=None, description='备注')


class CreateAIModelParam(AIModelSchemaBase):
    """创建 AI 模型参数"""


class UpdateAIModelParam(AIModelSchemaBase):
    """更新 AI 模型参数"""


class DeleteAIModelParam(SchemaBase):
    """删除 AI 模型参数"""

    pks: list[int] = Field(description='模型 ID 列表')


class GetAIModelDetail(AIModelSchemaBase):
    """AI 模型详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_time: datetime
    updated_time: datetime | None = None
