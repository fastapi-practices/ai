from dataclasses import dataclass
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
    context_start_index: int


@dataclass(frozen=True, slots=True)
class ChatRunContext:
    """协议运行上下文，核心聊天流程只读取通用字段"""

    conversation_id: str
    forwarded_props: AIChatForwardedPropsParam
    protocol_context: Any


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
    has_function_tool_sources: bool


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """单个构建器的产出"""

    capability: AbstractCapability[Any] | None
    introduces_builtin_tool: bool = False
    introduces_function_tool_source: bool = False


@dataclass(frozen=True, slots=True)
class AppendStrategy:
    """普通追加：新对话或续轮在 base_message_index 起追加"""

    base_message_index: int

    async def apply(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        payload_messages: list[dict],
        forwarded_props: AIChatForwardedPropsParam,
    ) -> None:
        """
        追加消息到对话末尾

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param payload_messages: 待落库消息列表
        :param forwarded_props: 聊天扩展参数
        :return:
        """
        if not payload_messages:
            return

        from backend.plugin.ai.crud.crud_message import ai_message_dao

        await ai_message_dao.bulk_create(
            db,
            [
                {
                    'conversation_id': conversation_id,
                    'provider_id': forwarded_props.provider_id,
                    'model_id': forwarded_props.model_id,
                    'message_index': self.base_message_index + offset,
                    'message': message,
                }
                for offset, message in enumerate(payload_messages)
            ],
        )


@dataclass(frozen=True, slots=True)
class ReplaceRangeStrategy:
    """替换指定行 ID 范围：AI 回复重生成 / 用户消息编辑"""

    row_ids: tuple[int, ...]
    start_index: int
    end_index: int

    async def apply(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        payload_messages: list[dict],
        forwarded_props: AIChatForwardedPropsParam,
    ) -> None:
        """
        按行 ID 替换并按需扩缩消息序列

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param payload_messages: 待落库消息列表
        :param forwarded_props: 聊天扩展参数
        :return:
        """
        replace_count = self.end_index - self.start_index + 1
        shared_count = min(replace_count, len(payload_messages))

        from backend.plugin.ai.crud.crud_message import ai_message_dao

        for index in range(shared_count):
            await ai_message_dao.update(
                db,
                self.row_ids[index],
                {
                    'provider_id': forwarded_props.provider_id,
                    'model_id': forwarded_props.model_id,
                    'message_index': self.start_index + index,
                    'message': payload_messages[index],
                },
            )

        if len(payload_messages) < replace_count:
            await ai_message_dao.delete_message_index_range(
                db,
                conversation_id,
                self.start_index + len(payload_messages),
                self.end_index,
            )
            await ai_message_dao.update_message_indexes_offset(
                db,
                conversation_id,
                self.end_index + 1,
                len(payload_messages) - replace_count,
            )
            return
        if len(payload_messages) == replace_count:
            return

        await ai_message_dao.update_message_indexes_offset(
            db,
            conversation_id,
            self.end_index + 1,
            len(payload_messages) - replace_count,
        )
        extras = payload_messages[replace_count:]
        await ai_message_dao.bulk_create(
            db,
            [
                {
                    'conversation_id': conversation_id,
                    'provider_id': forwarded_props.provider_id,
                    'model_id': forwarded_props.model_id,
                    'message_index': self.end_index + 1 + offset,
                    'message': message,
                }
                for offset, message in enumerate(extras)
            ],
        )


@dataclass(frozen=True, slots=True)
class InsertBeforeStrategy:
    """在指定 index 前插入：基于用户消息重新生成且保留后续消息"""

    insert_before_index: int
    base_message_index: int

    async def apply(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        payload_messages: list[dict],
        forwarded_props: AIChatForwardedPropsParam,
    ) -> None:
        """
        先抬升后续消息的 index，再在 base_message_index 起插入

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param payload_messages: 待落库消息列表
        :param forwarded_props: 聊天扩展参数
        :return:
        """
        if not payload_messages:
            return

        from backend.plugin.ai.crud.crud_message import ai_message_dao

        await ai_message_dao.update_message_indexes_offset(
            db,
            conversation_id,
            self.insert_before_index,
            len(payload_messages),
        )
        await ai_message_dao.bulk_create(
            db,
            [
                {
                    'conversation_id': conversation_id,
                    'provider_id': forwarded_props.provider_id,
                    'model_id': forwarded_props.model_id,
                    'message_index': self.base_message_index + offset,
                    'message': message,
                }
                for offset, message in enumerate(payload_messages)
            ],
        )


@dataclass(frozen=True, slots=True)
class CompletionPersistence:
    """聊天结果持久化上下文"""

    conversation_id: str
    user_id: int
    forwarded_props: AIChatForwardedPropsParam
    conversation: AIConversation | None
    title: str
    result_offset: int
    strategy: AppendStrategy | ReplaceRangeStrategy | InsertBeforeStrategy
