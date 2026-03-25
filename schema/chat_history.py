from datetime import datetime

from pydantic import ConfigDict, Field

from backend.common.schema import SchemaBase
from backend.plugin.ai.schema.chat import GetAIChatMessageDetail


class AIChatHistorySchemaBase(SchemaBase):
    """聊天历史基础模型"""

    conversation_id: str = Field(description='会话 ID')
    title: str = Field(description='会话标题')
    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    user_id: int = Field(description='用户 ID')
    pinned_time: datetime | None = Field(default=None, description='置顶时间')


class CreateAIChatHistoryParam(AIChatHistorySchemaBase):
    """创建聊天历史参数"""


class UpdateAIChatHistoryParam(AIChatHistorySchemaBase):
    """更新聊天历史参数"""


class UpdateAIChatConversationParam(SchemaBase):
    """更新聊天话题参数"""

    title: str = Field(description='会话标题')


class UpdateAIChatConversationPinParam(SchemaBase):
    """更新聊天话题置顶状态参数"""

    is_pinned: bool = Field(description='是否置顶')


class DeleteAIChatMessageResult(SchemaBase):
    """删除聊天消息结果"""

    deleted_conversation: bool = Field(description='是否删除了整个话题')
    remaining_message_count: int = Field(description='剩余消息数量')


class GetAIChatConversationItem(SchemaBase):
    """聊天历史详情基础信息"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='聊天历史 ID')
    conversation_id: str = Field(description='会话 ID')
    title: str = Field(description='会话标题')
    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')
    is_pinned: bool = Field(description='是否置顶')
    message_count: int = Field(description='消息数量')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class GetAIChatConversationListItem(SchemaBase):
    """聊天历史列表项"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='聊天历史 ID')
    conversation_id: str = Field(description='会话 ID')
    title: str = Field(description='会话标题')
    is_pinned: bool = Field(description='是否置顶')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class GetAIChatConversationDetail(GetAIChatConversationItem):
    """聊天历史详情"""

    messages: list[GetAIChatMessageDetail] = Field(description='对话消息列表')
