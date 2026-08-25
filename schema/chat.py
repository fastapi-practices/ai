from pydantic import Field

from backend.plugin.ai.protocol.default_schema import AIChatInputMessageParam
from backend.plugin.ai.protocol.schema import AIChatSchemaBase


class AIChatModelSelectParam(AIChatSchemaBase):
    """聊天模型选择参数"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')


class AIChatForwardedPropsParam(AIChatModelSelectParam):
    """对话模型参数"""


class AIChatRequestBase(AIChatSchemaBase):
    """聊天请求基础参数"""

    conversation_id: str | None = Field(default=None, description='对话 ID，不传则后端自动生成')
    forwarded_props: AIChatForwardedPropsParam = Field(description='聊天模型参数')


class AIChatCompletionParam(AIChatRequestBase):
    """聊天参数"""

    messages: list[AIChatInputMessageParam] = Field(min_length=1, description='当前轮输入消息列表')


class AIChatRegenerateParam(AIChatRequestBase):
    """重生成参数"""

    content: str | None = Field(default=None, description='重发时覆盖最后一条用户消息内容，空值保持原文')
