from datetime import datetime

from pydantic import ConfigDict, Field, model_validator

from backend.common.enums import StatusType
from backend.common.schema import SchemaBase


class AIModelSchemaBase(SchemaBase):
    """AI 模型基础模型"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    status: StatusType = Field(description='状态')
    context_max_part_chars: int | None = Field(
        default=None,
        ge=1,
        description='单个模型响应文本或工具调用参数超过此字符数时保留首尾并裁剪中间内容，空值表示不裁剪',
    )
    context_max_messages: int | None = Field(
        default=None,
        ge=1,
        description='发送模型前达到此消息数量时裁剪较早消息，空值表示不裁剪',
    )
    context_keep_messages: int = Field(
        default=60,
        ge=0,
        description='裁剪后保留的最近消息数量，首条用户消息和完整工具调用链会额外保留',
    )
    context_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description='上下文容量告警使用的最大 token 数量，空值关闭容量告警',
    )
    remark: str | None = Field(default=None, description='备注')

    @model_validator(mode='after')
    def validate_context_message_window(self) -> 'AIModelSchemaBase':
        """校验上下文消息窗口配置"""
        if self.context_max_messages is not None and self.context_keep_messages >= self.context_max_messages:
            raise ValueError('上下文保留消息数量必须小于最大消息数量')
        return self


class CreateAIModelParam(AIModelSchemaBase):
    """创建 AI 模型参数"""


class CreateAIModelsParam(SchemaBase):
    """批量创建 AI 模型参数"""

    items: list[CreateAIModelParam] = Field(max_length=200, description='模型列表')


class UpdateAIModelParam(AIModelSchemaBase):
    """更新 AI 模型参数"""


class DeleteAIModelParam(SchemaBase):
    """删除 AI 模型参数"""

    pks: list[int] = Field(description='模型 ID 列表')


class GetAIModelDetail(AIModelSchemaBase):
    """AI 模型详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='模型 ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(default=None, description='更新时间')
