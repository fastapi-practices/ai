from datetime import datetime

from ag_ui.core import MessagesSnapshotEvent
from pydantic import ConfigDict, Field

from backend.common.schema import SchemaBase


class AIConversationSchemaBase(SchemaBase):
    """AI 对话基础模型"""

    conversation_id: str = Field(description='对话 ID')
    title: str = Field(description='对话标题')
    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    user_id: int = Field(description='用户 ID')
    pinned_time: datetime | None = Field(default=None, description='置顶时间')
    context_start_message_id: int | None = Field(default=None, description='上下文起始消息 ID')
    context_cleared_time: datetime | None = Field(default=None, description='上下文清除时间')


class CreateAIConversationParam(AIConversationSchemaBase):
    """创建对话参数"""


class UpdateAIConversationParam(AIConversationSchemaBase):
    """更新对话参数"""


class UpdateAIConversationTitleParam(SchemaBase):
    """更新对话标题参数"""

    title: str = Field(description='对话标题')


class UpdateAIConversationPinnedParam(SchemaBase):
    """更新对话置顶状态参数"""

    is_pinned: bool = Field(description='是否置顶')


class GetAIConversationListItem(SchemaBase):
    """对话列表项"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='ID')
    conversation_id: str = Field(description='对话 ID')
    title: str = Field(description='对话标题')
    is_pinned: bool = Field(description='是否置顶')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class GetAIConversationDetail(SchemaBase):
    """对话详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='ID')
    conversation_id: str = Field(description='对话 ID')
    title: str = Field(description='对话标题')
    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    context_start_message_id: int | None = Field(default=None, description='上下文起始消息 ID')
    context_cleared_time: datetime | None = Field(default=None, description='上下文清除时间')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')
    messages_snapshot: MessagesSnapshotEvent = Field(description='对话消息快照')
