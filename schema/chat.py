from typing import Any, TypeAlias

from pydantic import ConfigDict, Field, HttpUrl

from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import (
    AIChatAttachmentSourceType,
    AIChatAttachmentType,
    AIChatMessageRoleType,
    AIChatOutputModeType,
)


class AIChatRequestBase(SchemaBase):
    """聊天请求基础模型"""

    model_config = ConfigDict(extra='forbid')

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
    extra_body: object | None = Field(default=None, description='发送给模型的额外请求体')
    parallel_tool_calls: bool | None = Field(default=True, description='是否允许并行工具调用')
    include_thinking: bool = Field(default=False, description='是否返回模型思考链')
    reasoning_effort: str | None = Field(default=None, description='模型推理强度')
    enable_builtin_tools: bool = Field(default=True, description='是否启用内置工具')
    mcp_ids: list[int] | None = Field(default=None, description='启用的 MCP ID 列表')
    output_mode: AIChatOutputModeType = Field(default=AIChatOutputModeType.text, description='输出模式')
    output_schema: dict[str, Any] | None = Field(default=None, description='结构化输出 JSON Schema')
    output_schema_name: str | None = Field(default=None, description='结构化输出名称')
    output_schema_description: str | None = Field(default=None, description='结构化输出说明')
    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='聊天模型')


class AIChatAttachmentParam(SchemaBase):
    """聊天附件参数"""

    type: AIChatAttachmentType = Field(description='附件类型')
    source_type: AIChatAttachmentSourceType = Field(description='附件来源类型')
    url: HttpUrl | None = Field(default=None, description='附件 URL')
    content: str | None = Field(default=None, description='Base64 编码内容')
    media_type: str | None = Field(default=None, description='附件媒体类型')


class CreateAIChatParam(AIChatRequestBase):
    """普通发送聊天参数"""

    conversation_id: str | None = Field(default=None, description='会话 ID，不传则创建新会话')
    user_prompt: str = Field(description='用户提示词')
    attachments: list[AIChatAttachmentParam] | None = Field(default=None, description='聊天附件')


class EditAIChatParam(CreateAIChatParam):
    """编辑重发聊天参数"""

    conversation_id: str = Field(description='会话 ID')
    edit_message_id: int = Field(description='编辑并重发的用户消息 ID')


class RegenerateAIChatParam(AIChatRequestBase):
    """重新生成聊天参数"""

    conversation_id: str = Field(description='会话 ID')
    regenerate_message_id: int = Field(description='需重新生成的 AI 消息数据库 ID')


AIChatParam: TypeAlias = CreateAIChatParam | EditAIChatParam | RegenerateAIChatParam


class UpdateAIChatMessageParam(SchemaBase):
    """更新聊天消息参数"""

    content: str = Field(description='消息内容')


class GetAIChatMessageDetail(SchemaBase):
    """AI 聊天消息详情"""

    message_id: int | None = Field(default=None, description='消息 ID')
    conversation_id: str | None = Field(default=None, description='会话 ID')
    message_index: int = Field(description='展示消息索引')
    role: AIChatMessageRoleType = Field(description='消息角色')
    timestamp: str = Field(description='消息时间')
    content: str = Field(description='消息内容')
    is_error: bool = Field(default=False, description='是否为错误消息')
    error_message: str | None = Field(default=None, description='错误详情')
    structured_data: Any | None = Field(default=None, description='结构化输出数据')
