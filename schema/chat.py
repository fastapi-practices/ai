from typing import Any, Literal

from pydantic import Field

from backend.plugin.ai.enums import AIChatGenerationType, AIChatThinkingType, AIWebSearchType
from backend.plugin.ai.protocol.default_schema import AIChatInputMessageParam
from backend.plugin.ai.protocol.schema import AIChatSchemaBase


class AIChatModelSelectParam(AIChatSchemaBase):
    """聊天模型选择参数"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='模型 ID')


class AIChatThinkingParam(AIChatSchemaBase):
    """聊天模型思考参数"""

    thinking: bool | AIChatThinkingType | None = Field(default=None, description='模型思考模式')


class AIChatRuntimeParam(AIChatSchemaBase):
    """聊天运行时控制参数"""

    enable_builtin_tools: bool = Field(default=True, description='是否启用项目内置工具')
    mcp_ids: list[int] | None = Field(default=None, description='启用的 MCP ID 列表')
    web_search: AIWebSearchType = Field(default=AIWebSearchType.off, description='网络搜索模式')


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


class AIChatImageGenerationParam(AIChatSchemaBase):
    """聊天图片生成参数"""

    image_action: Literal['generate', 'edit', 'auto'] | None = Field(default=None, description='图片生成动作')
    image_background: Literal['transparent', 'opaque', 'auto'] | None = Field(
        default=None,
        description='图片背景类型',
    )
    image_input_fidelity: Literal['high', 'low'] | None = Field(default=None, description='图片编辑输入保真度')
    image_moderation: Literal['auto', 'low'] | None = Field(default=None, description='图片生成审核强度')
    image_model: str | None = Field(default=None, description='图片生成底层模型')
    image_output_compression: int | None = Field(default=None, ge=0, le=100, description='图片输出压缩质量')
    image_output_format: Literal['png', 'webp', 'jpeg'] | None = Field(default=None, description='图片输出格式')
    image_partial_images: int | None = Field(default=None, ge=0, le=3, description='流式图片中间结果数量')
    image_quality: Literal['low', 'medium', 'high', 'auto'] | None = Field(default=None, description='图片生成质量')
    image_size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'] | None = Field(
        default=None,
        description='图片尺寸',
    )
    image_aspect_ratio: Literal['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'] | None = Field(
        default=None, description='图片宽高比'
    )


class AIChatOutputParam(AIChatSchemaBase):
    """聊天输出控制参数"""

    generation_type: AIChatGenerationType = Field(default=AIChatGenerationType.text, description='生成类型')


class AIChatForwardedPropsParam(
    AIChatModelSelectParam,
    AIChatThinkingParam,
    AIChatRuntimeParam,
    AIChatModelSettingsParam,
    AIChatImageGenerationParam,
    AIChatOutputParam,
):
    """对话扩展参数"""


class AIChatRequestBase(AIChatSchemaBase):
    """聊天请求基础参数"""

    conversation_id: str | None = Field(default=None, description='对话 ID，不传则后端自动生成')
    forwarded_props: AIChatForwardedPropsParam = Field(description='聊天扩展参数')


class AIChatCompletionParam(AIChatRequestBase):
    """聊天参数"""

    messages: list[AIChatInputMessageParam] = Field(min_length=1, description='当前轮输入消息列表')


class AIChatRegenerateParam(AIChatRequestBase):
    """重生成参数"""
