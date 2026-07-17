from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, TypeAlias

import anyio

from ag_ui.core import BaseEvent, RunAgentInput, RunErrorEvent
from pydantic_ai import Agent, AgentRunResult, BinaryImage, ModelRequest, ModelResponse, capture_run_messages
from pydantic_ai.ui import NativeEvent
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.responses import StreamingResponse

from backend.plugin.ai.dataclasses import ChatAgentDeps

ChatModelMessage: TypeAlias = ModelRequest | ModelResponse
ChatAgentOutput: TypeAlias = BinaryImage | str
ChatAgent: TypeAlias = Agent[ChatAgentDeps, ChatAgentOutput]


class _StreamLifecycle:
    """流式回调生命周期"""

    def __init__(
        self,
        *,
        on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]],
        on_run_error: Callable[[str, list[ChatModelMessage]], Awaitable[None]],
        on_interrupted: Callable[[list[ChatModelMessage]], Awaitable[None]],
    ) -> None:
        self.on_complete = on_complete
        self.on_run_error = on_run_error
        self.on_interrupted = on_interrupted
        self.run_finished = False
        self.error_message: str | None = None

    async def complete(self, result: AgentRunResult[Any]) -> None:
        """幂等执行完成回调"""
        if self.run_finished or self.error_message is not None:
            return
        try:
            await self.on_complete(result)
        except Exception as exc:
            # 完成回调失败时走错误终态，避免再被当成客户端中断导致 pending 锁死或状态打架
            if self.error_message is None:
                self.error_message = str(exc) or '完成回调失败'
            raise
        else:
            self.run_finished = True

    def record_error(self, message: str) -> None:
        """记录首个运行错误，等待原生消息状态完成收敛"""
        if self.run_finished or self.error_message is not None:
            return
        self.error_message = message

    async def finalize(self, messages: list[ChatModelMessage]) -> None:
        """在底层流关闭后持久化失败或中断状态"""
        if self.run_finished:
            return
        try:
            if self.error_message is not None:
                await self.on_run_error(self.error_message, messages)
            else:
                await self.on_interrupted(messages)
        finally:
            # 无论落库成败都标记结束，防止重复 finalize
            self.run_finished = True


def _extract_current_run_messages(
    *,
    captured_messages: Sequence[ChatModelMessage],
    message_history: Sequence[ChatModelMessage],
) -> list[ChatModelMessage]:
    """
    提取当前轮原生消息

    capture_run_messages 仅捕获本 run；优先按 run_id 过滤。
    无 run_id 时回退为全部捕获结果，避免中断落库丢消息。
    """
    if not captured_messages:
        return []
    history_run_ids = {message.run_id for message in message_history if message.run_id is not None}
    current_run_id = next(
        (
            message.run_id
            for message in reversed(captured_messages)
            if message.run_id is not None and message.run_id not in history_run_ids
        ),
        None,
    )
    if current_run_id is None:
        return list(captured_messages)
    return [message for message in captured_messages if message.run_id == current_run_id]


async def _settle_event_streams(
    *,
    event_stream: AsyncIterator[BaseEvent],
    native_stream: AsyncIterator[NativeEvent],
    stream_error: BaseException | None,
) -> None:
    """先停止原生流，再耗尽官方 AG-UI 转换器的尾部事件"""
    # shield：任务取消时仍完成流关闭与尾部耗尽，避免悬挂连接
    with anyio.CancelScope(shield=True):
        try:
            athrow = getattr(native_stream, 'athrow', None) if stream_error is not None else None
            if athrow is not None:
                try:
                    await athrow(stream_error)
                except BaseException:
                    # 原始流异常优先，关闭异常不能覆盖它
                    pass
            else:
                aclose = getattr(native_stream, 'aclose', None)
                if aclose is not None:
                    await aclose()
        finally:
            async for _ in event_stream:
                pass


async def _stream_with_lifecycle(
    *,
    event_stream: AsyncIterator[BaseEvent],
    native_stream: AsyncIterator[NativeEvent],
    message_history: Sequence[ChatModelMessage],
    lifecycle: _StreamLifecycle,
    encode_event: Callable[[BaseEvent], str],
    on_finish: Callable[[], Awaitable[None]] | None,
) -> AsyncIterator[str]:
    """消费事件流并保证中断清理"""
    current_run_messages: list[ChatModelMessage] = []
    stream_error: BaseException | None = None
    try:
        with capture_run_messages() as captured_messages:
            try:
                async for event in event_stream:
                    if isinstance(event, RunErrorEvent):
                        lifecycle.record_error(event.message or '')
                    yield encode_event(event)
            except BaseException as exc:
                stream_error = exc
                raise
            finally:
                try:
                    await _settle_event_streams(
                        event_stream=event_stream,
                        native_stream=native_stream,
                        stream_error=stream_error,
                    )
                finally:
                    current_run_messages.extend(
                        _extract_current_run_messages(
                            captured_messages=captured_messages,
                            message_history=message_history,
                        )
                    )
    finally:
        # shield：任务取消时仍完成落库/回调，避免状态与资源不一致
        with anyio.CancelScope(shield=True):
            try:
                await lifecycle.finalize(current_run_messages)
            finally:
                if on_finish:
                    await on_finish()


def build_streaming_response(
    *,
    user_id: int,
    agent: ChatAgent,
    run_input: RunAgentInput,
    accept: str | None,
    message_history: list[ChatModelMessage],
    on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]],
    on_run_error: Callable[[str, list[ChatModelMessage]], Awaitable[None]],
    on_interrupted: Callable[[list[ChatModelMessage]], Awaitable[None]],
    on_finish: Callable[[], Awaitable[None]] | None = None,
) -> StreamingResponse:
    """
    运行聊天代理并返回流式响应

    :param user_id: 用户 ID
    :param agent: 聊天代理
    :param run_input: 运行参数
    :param accept: Accept 请求头
    :param message_history: 消息历史
    :param on_complete: 完成回调
    :param on_run_error: 运行失败回调
    :param on_interrupted: 运行中断回调
    :param on_finish: 流结束回调
    :return:
    """
    adapter = AGUIAdapter(
        agent=agent,
        run_input=run_input,
        accept=accept,
        allow_uploaded_files=True,
        preserve_file_data=True,
    )
    lifecycle = _StreamLifecycle(
        on_complete=on_complete,
        on_run_error=on_run_error,
        on_interrupted=on_interrupted,
    )
    native_stream = adapter.run_stream_native(
        deps=ChatAgentDeps(user_id=user_id),
        message_history=message_history,
    )
    event_stream_handler = adapter.build_event_stream()
    event_stream = event_stream_handler.transform_stream(
        native_stream,
        on_complete=lifecycle.complete,
    )
    response = StreamingResponse(
        _stream_with_lifecycle(
            event_stream=event_stream,
            native_stream=native_stream,
            message_history=message_history,
            lifecycle=lifecycle,
            encode_event=event_stream_handler.encode_event,
            on_finish=on_finish,
        ),
        headers=event_stream_handler.response_headers,
        media_type=event_stream_handler.content_type,
    )
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response
