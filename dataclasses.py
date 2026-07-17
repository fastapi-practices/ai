from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.native_tools import AbstractNativeTool
from sqlalchemy.ext.asyncio import AsyncSession

from backend.plugin.ai.model.conversation import AIConversation
from backend.plugin.ai.model.message import AIMessage
from backend.plugin.ai.providers.base import ProviderAdapter
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


@dataclass(slots=True)
class ChatAgentDeps:
    """聊天代理依赖"""

    user_id: int


@dataclass(slots=True)
class ChatConversationState:
    """聊天上下文状态"""

    conversation: AIConversation | None
    message_rows: list[AIMessage]
    model_messages: list[ModelRequest | ModelResponse]
    row_model_message_ranges: list[tuple[int, int]]
    context_start_index: int


@dataclass(frozen=True, slots=True)
class ChatRunContext:
    """协议运行上下文，核心聊天流程只读取通用字段"""

    conversation_id: str
    forwarded_props: AIChatForwardedPropsParam
    protocol_context: Any


@dataclass(frozen=True, slots=True)
class ContextManagementPolicy:
    """上下文管理策略"""

    max_part_chars: int | None
    max_messages: int | None
    keep_messages: int
    max_tokens: int | None
    warning_threshold: float


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    """能力构建上下文"""

    db: AsyncSession
    adapter: ProviderAdapter
    forwarded_props: AIChatForwardedPropsParam
    supports_tools: bool
    supported_native_tools: frozenset[type[AbstractNativeTool]]
    supports_image_output: bool
    has_builtin_tools: bool
    context_management: ContextManagementPolicy


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """单个构建器的产出"""

    capability: AbstractCapability[Any] | None
    introduces_builtin_tool: bool = False
    introduces_function_tool_source: bool = False


@dataclass(frozen=True, slots=True)
class CompletionPersistenceContext:
    """普通聊天完成持久化上下文"""

    conversation_id: str
    user_id: int
    forwarded_props: AIChatForwardedPropsParam
    title: str
    assistant_message_id: int | None = None


@dataclass(frozen=True, slots=True)
class RegenerationPersistenceContext:
    """重生成完成持久化上下文"""

    conversation_id: str
    user_id: int
    forwarded_props: AIChatForwardedPropsParam
    expected_message_versions: tuple[tuple[int, datetime | None], ...] = ()
    expected_context_start_message_id: int | None = None
    expected_context_cleared_time: datetime | None = None
    assistant_message_id: int | None = None
    insert_before_index: int | None = None
    replace_start_index: int | None = None
    replace_end_index: int | None = None
