from dataclasses import dataclass, field
from typing import Any

from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType


@dataclass(frozen=True, slots=True)
class AIInvocationContext:
    """AI 调用策略上下文"""

    user_id: int
    provider_id: int
    provider_type: AIProviderType
    provider_name: str
    model_pk: int
    model_id: str
    mcp_ids: tuple[int, ...] = ()
    generation_type: AIChatGenerationType | None = None
    conversation_id: str | None = None


@dataclass(frozen=True, slots=True)
class AIInvocationResult:
    """AI 调用结果策略上下文"""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_count: int | None = None
    tool_call_count: int | None = None
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    cache_audio_read_tokens: int | None = None
    usage_details: dict[str, int] = field(default_factory=dict)
    raw_result: Any | None = None

    @classmethod
    def from_agent_result(cls, result: Any) -> 'AIInvocationResult':
        """
        从 Pydantic AI 调用结果构建策略结果

        :param result: Pydantic AI 调用结果
        :return:
        """
        usage = getattr(result, 'usage', None)
        if usage is None:
            return cls(raw_result=result)
        return cls(
            input_tokens=getattr(usage, 'input_tokens', None),
            output_tokens=getattr(usage, 'output_tokens', None),
            total_tokens=getattr(usage, 'total_tokens', None),
            request_count=getattr(usage, 'requests', None),
            tool_call_count=getattr(usage, 'tool_calls', None),
            cache_write_tokens=getattr(usage, 'cache_write_tokens', None),
            cache_read_tokens=getattr(usage, 'cache_read_tokens', None),
            input_audio_tokens=getattr(usage, 'input_audio_tokens', None),
            output_audio_tokens=getattr(usage, 'output_audio_tokens', None),
            cache_audio_read_tokens=getattr(usage, 'cache_audio_read_tokens', None),
            usage_details=dict(getattr(usage, 'details', {}) or {}),
            raw_result=result,
        )
