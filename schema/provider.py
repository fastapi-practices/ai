from datetime import datetime

from pydantic import ConfigDict, Field

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase


class AiProviderSchemaBase(SchemaBase):
    """供应商基础模型"""

    name: str = Field(description='供应商名称')
    type: int = Field(description='供应商类型（0OpenAI 1Anthropic 2Gemini）')
    api_key: str = Field(description='API Key')
    api_host: str = Field(description='API Host')
    status: StatusType = Field(description='状态')
    remark: str | None = Field(None, description='备注')


class CreateAiProviderParam(AiProviderSchemaBase):
    """创建供应商参数"""


class UpdateAiProviderParam(AiProviderSchemaBase):
    """更新供应商参数"""


class DeleteAiProviderParam(SchemaBase):
    """删除供应商参数"""

    pks: list[int] = Field(description='供应商 ID 列表')


class GetAiProviderDetail(AiProviderSchemaBase):
    """供应商详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_time: datetime
    updated_time: datetime | None = None
