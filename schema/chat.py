from typing import Any

from ag_ui.core import UserMessage
from pydantic import ConfigDict, Field

from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import (
    AIChatGenerationType,
    AIChatReasoningEffortType,
    AIWebSearchType,
)


class AIChatModelSelectParam(SchemaBase):
    """聊天模型选择参数"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')


class AIChatReasoningParam(SchemaBase):
    """聊天模型思考推理参数"""

    include_thinking: bool = Field(default=True, description='是否返回模型思考链')
    reasoning_effort: AIChatReasoningEffortType | None = Field(default=None, description='模型推理强度')


class AIChatRuntimeParam(SchemaBase):
    """聊天运行时控制参数"""

    enable_builtin_tools: bool = Field(default=True, description='是否启用项目内置工具')
    mcp_ids: list[int] | None = Field(default=None, description='启用的 MCP ID 列表')
    web_search: AIWebSearchType = Field(default=AIWebSearchType.builtin, description='网络搜索模式')


class AIChatModelSettingsParam(SchemaBase):
    """聊天模型采样参数"""

    max_tokens: int | None = Field(default=None, description='停止前最多可生成的 token 数')
    temperature: float | None = Field(default=1.0, description='模型生成文本的随机性')
    top_p: float | None = Field(default=None, description='模型生成文本的多样性')
    timeout: float | None = Field(default=None, description='覆盖客户端对请求的默认超时（单位：s）')
    seed: int | None = Field(default=None, description='用于模型的随机种子')
    presence_penalty: float | None = Field(default=None, description='根据新 token 是否出现在文本中来处罚')
    frequency_penalty: float | None = Field(default=None, description='根据新 token 目前在文本中的出现频率进行惩罚')
    logit_bias: dict[str, int] | None = Field(default=None, description='修改完成中出现指定标记的可能性')
    stop_sequences: list[str] | None = Field(default=None, description='这些序列会导致模型停止生成')
    extra_headers: dict[str, str] | None = Field(default=None, description='发送给模型的额外 Headers')
    extra_body: dict[str, Any] | None = Field(default=None, description='发送给模型的额外请求体')
    parallel_tool_calls: bool | None = Field(default=True, description='是否允许并行工具调用')


class AIChatOutputParam(SchemaBase):
    """聊天输出控制参数"""

    generation_type: AIChatGenerationType = Field(default=AIChatGenerationType.text, description='生成类型')


class AIChatForwardedPropsParam(
    AIChatModelSelectParam,
    AIChatReasoningParam,
    AIChatRuntimeParam,
    AIChatModelSettingsParam,
    AIChatOutputParam,
):
    """对话扩展参数"""

    model_config = ConfigDict(extra='forbid')


class AIChatRequestBase(SchemaBase):
    """聊天请求基础参数"""

    model_config = ConfigDict(extra='forbid')

    thread_id: str | None = Field(default=None, description='对话 ID，不传则后端自动生成')
    forwarded_props: AIChatForwardedPropsParam = Field(description='聊天扩展参数')


class AIChatCompletionParam(AIChatRequestBase):
    """聊天参数"""

    message: UserMessage = Field(description='当前轮用户消息')


class AIChatRegenerateParam(AIChatRequestBase):
    """重生成参数"""
