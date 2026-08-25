from typing import Annotated, Any, TypeAlias

from ag_ui.core import (
    ActivityMessage,
    AssistantMessage,
    AudioInputContent,
    DeveloperMessage,
    DocumentInputContent,
    ImageInputContent,
    MessagesSnapshotEvent,
    ReasoningMessage,
    SystemMessage,
    TextInputContent,
    ToolMessage,
    UserMessage,
    VideoInputContent,
)
from pydantic import Field

from backend.plugin.ai.protocol.schema import AIChatMessageMetaSchemaBase, AIChatSchemaBase

AIChatAgUiInputContentParam: TypeAlias = Annotated[
    TextInputContent | ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent,
    Field(discriminator='type'),
]


class AIChatAgUiUserMessageInput(UserMessage, AIChatSchemaBase):
    """AI 对话用户输入消息参数"""

    content: str | list[AIChatAgUiInputContentParam] = Field(description='用户输入内容')


AIChatAgUiInputMessageParam: TypeAlias = AIChatAgUiUserMessageInput


class AIChatAgUiUserMessageDetail(UserMessage, AIChatMessageMetaSchemaBase):
    """AI 对话用户消息详情"""

    content: str | list[AIChatAgUiInputContentParam] = Field(description='用户消息内容')


class AIChatAgUiDeveloperMessageDetail(DeveloperMessage, AIChatMessageMetaSchemaBase):
    """AI 对话开发者消息详情"""


class AIChatAgUiAssistantMessageDetail(AssistantMessage, AIChatMessageMetaSchemaBase):
    """AI 对话助手消息详情"""


class AIChatAgUiSystemMessageDetail(SystemMessage, AIChatMessageMetaSchemaBase):
    """AI 对话系统消息详情"""


class AIChatAgUiToolMessageDetail(ToolMessage, AIChatMessageMetaSchemaBase):
    """AI 对话工具消息详情"""


class AIChatAgUiReasoningMessageDetail(ReasoningMessage, AIChatMessageMetaSchemaBase):
    """AI 对话推理消息详情"""


class AIChatAgUiActivityMessageDetail(ActivityMessage, AIChatMessageMetaSchemaBase):
    """AI 对话活动消息详情"""

    content: dict[str, Any] = Field(description='活动消息内容')


AIChatAgUiSnapshotMessageDetail: TypeAlias = Annotated[
    AIChatAgUiUserMessageDetail
    | AIChatAgUiDeveloperMessageDetail
    | AIChatAgUiAssistantMessageDetail
    | AIChatAgUiSystemMessageDetail
    | AIChatAgUiToolMessageDetail
    | AIChatAgUiReasoningMessageDetail
    | AIChatAgUiActivityMessageDetail,
    Field(discriminator='role'),
]


class AIChatAgUiMessagesSnapshotDetail(MessagesSnapshotEvent, AIChatSchemaBase):
    """AI 对话消息快照详情"""

    messages: list[AIChatAgUiSnapshotMessageDetail] = Field(description='消息快照列表')
