from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from ag_ui.core import RunAgentInput
from pydantic_ai import AgentRunResult, ModelMessage
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.database.db import uuid4_str
from backend.plugin.ai.dataclasses import ChatRunContext
from backend.plugin.ai.protocol.ag_ui.event_stream import build_streaming_response
from backend.plugin.ai.protocol.ag_ui.request_decoder import decode_input_messages as decode_ag_ui_input_messages
from backend.plugin.ai.protocol.ag_ui.snapshot_builder import serialize_messages_to_snapshot as serialize_ag_ui_snapshot
from backend.plugin.ai.protocol.base import ChatAgent, ChatModelMessage
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


class AgUiChatProtocolAdapter:
    """AG-UI 聊天协议适配器"""

    name = 'ag_ui'

    @staticmethod
    def decode_input_messages(*, messages: Sequence[Any]) -> list[ModelMessage]:
        """
        解析当前轮输入消息列表

        :param messages: 输入消息列表
        :return:
        """
        return decode_ag_ui_input_messages(messages=messages)

    @staticmethod
    def sanitize_input_messages(
        *,
        agent: ChatAgent,
        run_context: ChatRunContext,
        messages: Sequence[ModelMessage],
    ) -> list[ModelMessage]:
        """
        按 AG-UI 规则清洗当前轮输入消息

        :param agent: 聊天代理
        :param run_context: 协议运行上下文
        :param messages: 模型消息列表
        :return:
        """
        adapter = AGUIAdapter(
            agent=agent,
            run_input=cast('RunAgentInput', run_context.protocol_context),
            allow_uploaded_files=True,
            preserve_file_data=True,
        )
        return adapter.sanitize_messages(messages)

    @staticmethod
    def build_run_context(
        *,
        conversation_id: str | None,
        forwarded_props: AIChatForwardedPropsParam,
        default_conversation_id: str | None = None,
        expected_conversation_id: str | None = None,
    ) -> ChatRunContext:
        """
        解析并补全运行参数

        :param conversation_id: 对话 ID
        :param forwarded_props: 聊天扩展参数
        :param default_conversation_id: 默认对话 ID
        :param expected_conversation_id: 期望对话 ID
        :return:
        """
        resolved_conversation_id = conversation_id or default_conversation_id or uuid4_str()
        run_input = RunAgentInput.model_validate({
            'thread_id': resolved_conversation_id,
            'run_id': uuid4_str(),
            'parent_run_id': None,
            'state': {},
            'messages': [],
            'tools': [],
            'context': [],
            'forwarded_props': forwarded_props.model_dump(),
        })

        if expected_conversation_id is not None and run_input.thread_id != expected_conversation_id:
            raise errors.RequestError(msg='请求体中的对话 ID 与路径不一致')

        return ChatRunContext(
            conversation_id=run_input.thread_id,
            forwarded_props=AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {}),
            protocol_context=run_input,
        )

    @staticmethod
    def build_streaming_response(
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
        运行聊天代理并返回 AG-UI 流式响应

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
        return build_streaming_response(
            user_id=user_id,
            agent=agent,
            run_input=cast('RunAgentInput', run_context.protocol_context),
            accept=accept,
            message_history=message_history,
            on_complete=on_complete,
            on_run_error=on_run_error,
            on_finish=on_finish,
        )

    @staticmethod
    def serialize_messages_to_snapshot(
        messages: Sequence[ModelMessage],
        *,
        conversation_id: str | None = None,
        message_ids: Sequence[int | None] | None = None,
        provider_ids: Sequence[int | None] | None = None,
        model_ids: Sequence[str | None] | None = None,
        message_indexes: Sequence[int | None] | None = None,
    ) -> Any:
        """
        序列化模型消息为 AG-UI 快照

        :param messages: 模型消息列表
        :param conversation_id: 对话 ID
        :param message_ids: 持久化消息 ID 列表
        :param provider_ids: 供应商 ID 列表
        :param model_ids: 模型 ID 列表
        :param message_indexes: 持久化消息索引列表
        :return:
        """
        return serialize_ag_ui_snapshot(
            messages,
            conversation_id=conversation_id,
            message_ids=message_ids,
            provider_ids=provider_ids,
            model_ids=model_ids,
            message_indexes=message_indexes,
        )


ag_ui_chat_protocol_adapter = AgUiChatProtocolAdapter()
