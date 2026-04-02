from pydantic import Field

from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import (
    AIChatAttachmentSourceType,
    AIChatAttachmentType,
    AIMessageBlockType,
    AIMessageRoleType,
    AIMessageType,
)


class UpdateAIMessageParam(SchemaBase):
    """更新消息参数"""

    content: str = Field(description='消息内容')


class GetAIMessageBlockDetail(SchemaBase):
    """AI 消息内容块详情"""

    type: AIMessageBlockType = Field(description='内容块类型')
    text: str | None = Field(default=None, description='文本内容')
    file_type: AIChatAttachmentType | None = Field(default=None, description='文件类型')
    source_type: AIChatAttachmentSourceType | None = Field(default=None, description='文件来源类型')
    mime_type: str | None = Field(default=None, description='文件内容类型')
    name: str | None = Field(default=None, description='文件名称')
    url: str | None = Field(default=None, description='文件地址')


class GetAIMessageDetail(SchemaBase):
    """AI 消息详情"""

    message_id: int | None = Field(default=None, description='消息 ID')
    conversation_id: str | None = Field(default=None, description='对话 ID')
    message_index: int = Field(description='展示消息索引')
    role: AIMessageRoleType = Field(description='消息角色')
    message_type: AIMessageType = Field(default=AIMessageType.normal, description='消息类型')
    created_time: str = Field(description='消息时间')
    provider_id: int | None = Field(default=None, description='供应商 ID')
    model_id: str | None = Field(default=None, description='模型 ID')
    blocks: list[GetAIMessageBlockDetail] = Field(default_factory=list, description='消息内容块列表')
