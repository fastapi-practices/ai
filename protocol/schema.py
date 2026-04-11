from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from backend.common.schema import SchemaBase


class AIChatSchemaBase(SchemaBase):
    """
    AI Chat 接口模型基础配置

    为兼容协议小驼峰返回，chat 接口（请求/响应）统一使用小驼峰参数
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AIChatMessageMetaSchemaBase(AIChatSchemaBase):
    """AI 对话消息扩展元信息"""

    conversation_id: str | None = Field(default=None, description='对话 ID')
    persisted_message_id: int | None = Field(default=None, description='持久化消息 ID')
    provider_id: int | None = Field(default=None, description='供应商 ID')
    model_id: str | None = Field(default=None, description='模型 ID')
    message_type: Literal['normal', 'error'] | None = Field(default=None, description='消息类型')
    message_index: int | None = Field(default=None, description='消息索引')
    created_time: datetime | None = Field(default=None, description='创建时间')
