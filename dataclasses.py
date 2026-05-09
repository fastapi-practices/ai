from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import ModelRequest, ModelResponse, Tool
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
from sqlalchemy.ext.asyncio import AsyncSession

from backend.plugin.ai.model.conversation import AIConversation
from backend.plugin.ai.model.message import AIMessage
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


@dataclass(slots=True)
class ChatAgentDeps:
    """聊天代理依赖"""

    db: AsyncSession
    user_id: int


@dataclass(slots=True)
class ChatAgentParts:
    """聊天代理参数片段"""

    tools: list[Tool[Any]] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)
    toolsets: list[AbstractToolset[Any]] = field(default_factory=list)
    capabilities: list[AbstractCapability[Any]] = field(default_factory=list)


@dataclass(slots=True)
class ChatCompletionPersistence:
    """聊天结果持久化上下文"""

    conversation_id: str
    user_id: int
    forwarded_props: AIChatForwardedPropsParam
    conversation: AIConversation | None
    title: str
    replace_message_row_ids: list[int] | None
    replace_start_message_index: int | None
    replace_end_message_index: int | None
    insert_before_message_index: int | None
    base_message_index: int
    result_offset: int


@dataclass(slots=True)
class ChatConversationState:
    """聊天上下文状态"""

    conversation: AIConversation | None
    message_rows: list[AIMessage]
    model_messages: list[ModelRequest | ModelResponse]
    context_start_index: int
