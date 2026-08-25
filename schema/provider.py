from datetime import datetime

from pydantic import ConfigDict, Field, field_serializer

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.utils.api_key_ops import mask_api_key


class AIProviderSchemaBase(SchemaBase):
    """AI 供应商基础模型"""

    name: str = Field(description='供应商名称')
    type: AIProviderType = Field(description='供应商类型')
    api_key: str = Field(description='接口密钥')
    api_host: str = Field(description='接口地址')
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

    id: int = Field(description='供应商 ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(default=None, description='更新时间')

    @field_serializer('api_key')
    def serialize_api_key(self, api_key: str) -> str:
        """脱敏返回接口密钥"""
        return mask_api_key(api_key)


class GetAIProviderModelDetail(SchemaBase):
    """获取供应商模型详情"""

    id: str = Field(description='模型标识符')
    object: str | None = Field(default=None, description='模型对象类型')
    created: int | None = Field(default=None, description='模型创建时的 Unix 时间戳（以秒为单位）')
