from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeAlias

from ag_ui.core import BaseEvent, RunAgentInput, RunErrorEvent
from pydantic_ai import Agent, AgentRunResult, ModelRequest, ModelResponse
from pydantic_ai.ui.ag_ui import AGUIAdapter
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.plugin.ai.dataclasses import ChatAgentDeps

ChatModelMessage: TypeAlias = ModelRequest | ModelResponse


async def ag_ui_event_encoder(stream: AsyncIterator[BaseEvent]) -> AsyncIterator[str]:
    """
    AG-UI 事件流编码器

    :param stream: AG-UI 事件流
    :return:
    """
    async for event in stream:
        yield f'data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n'


def build_streaming_response(
    *,
    db: AsyncSession,
    user_id: int,
    agent: Agent,
    run_input: RunAgentInput,
    accept: str | None,
    message_history: list[ChatModelMessage],
    on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]],
    on_run_error: Callable[[RunErrorEvent], Awaitable[None]],
) -> StreamingResponse:
    """
    运行聊天代理并返回流式响应

    :param db: 数据库会话
    :param user_id: 用户 ID
    :param agent: 聊天代理
    :param run_input: 运行参数
    :param accept: Accept 请求头
    :param message_history: 消息历史
    :param on_complete: 完成回调
    :param on_run_error: 运行失败回调
    :return:
    """
    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream(
        deps=ChatAgentDeps(db=db, user_id=user_id),
        message_history=message_history,
        on_complete=on_complete,
    )

    async def stream_with_error_callback() -> AsyncIterator[BaseEvent]:
        error_handled = False
        async for event in event_stream:
            if isinstance(event, RunErrorEvent) and not error_handled:
                error_handled = True
                await on_run_error(event)
            yield event

    event_stream_handler = adapter.build_event_stream()
    response = StreamingResponse(
        ag_ui_event_encoder(stream_with_error_callback()),
        headers=event_stream_handler.response_headers,
        media_type=event_stream_handler.content_type,
    )
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response
