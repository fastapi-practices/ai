from typing import Any

from pydantic import Field

from backend.plugin.ai.enums import AIChatGenerationType, AIChatThinkingType, AIWebSearchType
from backend.plugin.ai.protocol.ag_ui.schema import AIChatAgUiInputMessageParam
from backend.plugin.ai.protocol.schema import AIChatSchemaBase


class AIChatModelSelectParam(AIChatSchemaBase):
    """聊天模型选择参数"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')


class AIChatThinkingParam(AIChatSchemaBase):
    """聊天模型思考参数"""

    thinking: bool | AIChatThinkingType | None = Field(
        default=None,
        description='模型思考模式，支持 true、false 或 minimal/low/medium/high/xhigh',
    )


class AIChatRuntimeParam(AIChatSchemaBase):
    """聊天运行时控制参数"""

    enable_builtin_tools: bool = Field(default=True, description='是否启用项目内置工具')
    mcp_ids: list[int] | None = Field(default=None, description='启用的 MCP ID 列表')
    web_search: AIWebSearchType = Field(default=AIWebSearchType.builtin, description='网络搜索模式')


class AIChatModelSettingsParam(AIChatSchemaBase):
    """聊天模型采样参数"""

    max_tokens: int | None = Field(default=None, description='停止前最多可生成的 token 数')
    temperature: float | None = Field(default=None, description='模型生成文本的随机性')
    top_p: float | None = Field(default=None, description='模型生成文本的多样性')
    timeout: float | None = Field(default=None, description='覆盖客户端对请求的默认超时（单位：s）')
    seed: int | None = Field(default=None, description='用于模型的随机种子')
    presence_penalty: float | None = Field(default=None, description='根据新 token 是否出现在文本中来处罚')
    frequency_penalty: float | None = Field(default=None, description='根据新 token 目前在文本中的出现频率进行惩罚')
    logit_bias: dict[str, int] | None = Field(default=None, description='修改完成中出现指定标记的可能性')
    stop_sequences: list[str] | None = Field(default=None, description='这些序列会导致模型停止生成')
    extra_headers: dict[str, str] | None = Field(default=None, description='发送给模型的额外 Headers')
    extra_body: dict[str, Any] | None = Field(default=None, description='发送给模型的额外请求体')
    parallel_tool_calls: bool | None = Field(default=None, description='是否允许并行工具调用')


class AIChatOutputParam(AIChatSchemaBase):
    """聊天输出控制参数"""

    generation_type: AIChatGenerationType = Field(default=AIChatGenerationType.text, description='生成类型')


class AIChatForwardedPropsParam(
    AIChatModelSelectParam,
    AIChatThinkingParam,
    AIChatRuntimeParam,
    AIChatModelSettingsParam,
    AIChatOutputParam,
):
    """对话扩展参数"""


class AIChatRequestBase(AIChatSchemaBase):
    """聊天请求基础参数"""

    conversation_id: str | None = Field(default=None, description='对话 ID，不传则后端自动生成')
    forwarded_props: AIChatForwardedPropsParam = Field(description='聊天扩展参数')


class AIChatCompletionParam(AIChatRequestBase):
    """聊天参数"""

    messages: list[AIChatAgUiInputMessageParam] = Field(min_length=1, description='当前轮输入消息列表')


class AIChatRegenerateParam(AIChatRequestBase):
    """重生成参数"""
