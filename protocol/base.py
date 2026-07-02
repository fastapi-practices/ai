from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, TypeAlias

from pydantic_ai import Agent, AgentRunResult, BinaryImage, ModelMessage, ModelRequest, ModelResponse
from starlette.responses import StreamingResponse

from backend.plugin.ai.dataclasses import ChatAgentDeps, ChatRunContext
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

ChatModelMessage: TypeAlias = ModelRequest | ModelResponse
ChatAgentOutput: TypeAlias = BinaryImage | str
ChatAgent: TypeAlias = Agent[ChatAgentDeps, ChatAgentOutput]


class ChatProtocolAdapter(Protocol):
    """聊天协议适配器"""

    name: str

    def decode_input_messages(self, *, messages: Sequence[Any]) -> list[ModelMessage]:
        """
        解析当前轮输入消息列表

        :param messages: 协议输入消息列表
        :return:
        """
        ...

    def sanitize_input_messages(
        self,
        *,
        agent: ChatAgent,
        run_context: ChatRunContext,
        messages: Sequence[ModelMessage],
    ) -> list[ModelMessage]:
        """
        清洗当前轮输入消息

        :param agent: 聊天代理
        :param run_context: 协议运行上下文
        :param messages: 模型消息列表
        :return:
        """
        ...

    def build_run_context(
        self,
        *,
        conversation_id: str | None,
        forwarded_props: AIChatForwardedPropsParam,
        default_conversation_id: str | None = None,
        expected_conversation_id: str | None = None,
    ) -> ChatRunContext:
        """
        构建协议运行上下文

        :param conversation_id: 对话 ID
        :param forwarded_props: 聊天扩展参数
        :param default_conversation_id: 默认对话 ID
        :param expected_conversation_id: 期望对话 ID
        :return:
        """
        ...

    def build_streaming_response(
        self,
        *,
        user_id: int,
        agent: ChatAgent,
        run_context: ChatRunContext,
        accept: str | None,
        message_history: list[ChatModelMessage],
        on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]],
        on_run_error: Callable[[str], Awaitable[None]],
        on_finish: Callable[[], Awaitable[None]] | None = None,
    ) -> StreamingResponse:
        """
        运行聊天代理并构建协议流式响应

        :param user_id: 用户 ID
        :param agent: 聊天代理
        :param run_context: 协议运行上下文
        :param accept: Accept 请求头
        :param message_history: 消息历史
        :param on_complete: 完成回调
        :param on_run_error: 运行失败回调
        :param on_finish: 流结束回调
        :return:
        """
        ...

    def serialize_messages_to_snapshot(
        self,
        messages: Sequence[ModelMessage],
        *,
        conversation_id: str | None = None,
        message_ids: Sequence[int | None] | None = None,
        provider_ids: Sequence[int | None] | None = None,
        model_ids: Sequence[str | None] | None = None,
        message_indexes: Sequence[int | None] | None = None,
    ) -> Any:
        """
        序列化模型消息为协议快照

        :param messages: 模型消息列表
        :param conversation_id: 对话 ID
        :param message_ids: 持久化消息 ID 列表
        :param provider_ids: 供应商 ID 列表
        :param model_ids: 模型 ID 列表
        :param message_indexes: 持久化消息索引列表
        :return:
        """
        ...
