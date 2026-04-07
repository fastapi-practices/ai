from dataclasses import dataclass

from pydantic_ai import ModelRequest, ModelResponse
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
