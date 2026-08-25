from pydantic import ConfigDict, Field

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.schema.default_model import GetAIDefaultModelDetail
from backend.plugin.ai.schema.model import GetAIModelDetail


class GetAIProviderModelOptionDetail(SchemaBase):
    """AI 供应商模型选项详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='供应商 ID')
    name: str = Field(description='供应商名称')
    type: AIProviderType = Field(description='供应商类型')
    status: StatusType = Field(description='状态')
    models: list[GetAIModelDetail] = Field(description='启用的模型列表')


class GetAIModelOptionsDetail(SchemaBase):
    """AI 模型选项详情"""

    providers: list[GetAIProviderModelOptionDetail] = Field(description='启用的供应商及模型列表')
    default_model: GetAIDefaultModelDetail | None = Field(default=None, description='默认助手模型配置')
