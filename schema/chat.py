from pydantic import Field

from backend.common.schema import SchemaBase


class AIChatSchemaBase(SchemaBase):
    """聊天基础模型"""

    max_tokens: int | None = Field(default=None, description='停止前最多可生成的 token 数')
    temperature: float | None = Field(default=1.0, description='模型生成文本的随机性')
    top_p: float | None = Field(default=1.0, description='模型生成文本的多样性')
    timeout: float | None = Field(default=None, description='覆盖客户端对请求的默认超时（单位：s）')
    parallel_tool_calls: bool | None = Field(default=True, description='是否允许并行工具调用')
    seed: int | None = Field(default=None, description='用于模型的随机种子')
    presence_penalty: float | None = Field(default=None, description='根据新 token 是否出现在文本中来处罚')
    frequency_penalty: float | None = Field(default=None, description='根据新 token 目前在文本中的出现频率进行惩罚')
    logit_bias: dict[str, int] | None = Field(default=None, description='修改完成中出现指定标记的可能性')
    stop_sequences: list[str] | None = Field(default=None, description='这些序列会导致模型停止生成')
    extra_headers: dict[str, str] | None = Field(default=None, description='发送给模型的额外 Headers')
    extra_body: object | None = Field(default=None, description='发送给模型的额外请求体')


class AIChat(AIChatSchemaBase):
    """聊天参数"""

    provider_id: int = Field(description='供应商 ID')
    model_id: str = Field(description='聊天模型')
    user_prompt: str = Field(description='用户提示词')
