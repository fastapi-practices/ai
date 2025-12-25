from datetime import datetime

from pydantic import ConfigDict, Field

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import AIProviderType


class AIProviderSchemaBase(SchemaBase):
    """AI 供应商基础模型"""

    name: str = Field(description='供应商名称')
    type: AIProviderType = Field(description='供应商类型（0OpenAI 1Anthropic 2Gemini）')
    api_key: str = Field(description='API Key')
    api_host: str = Field(description='API Host')
    status: StatusType = Field(description='状态')
    remark: str | None = Field(None, description='备注')


class CreateAIProviderParam(AIProviderSchemaBase):
    """创建 AI 供应商参数"""


class UpdateAIProviderParam(AIProviderSchemaBase):
    """更新 AI 供应商参数"""


class DeleteAIProviderParam(SchemaBase):
    """删除 AI 供应商参数"""

    pks: list[int] = Field(description='供应商 ID 列表')


class GetAIProviderDetail(AIProviderSchemaBase):
    """AI 供应商详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_time: datetime
    updated_time: datetime | None = None


class GetAIProviderModelDetail(SchemaBase):
    """获取供应商模型详情"""

    id: str = Field(description='模型标识符')
    object: str = Field(description='对象类型始终为 “model”')
    created: int = Field(description='模型创建时的 Unix 时间戳（以秒为单位）')
    owned_by: str = Field(description='拥有该模型的组织')
