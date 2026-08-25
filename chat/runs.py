from __future__ import annotations

import asyncio

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from pydantic_ai import CancellationToken

from backend.common.exception import errors
from backend.common.log import log
from backend.plugin.ai.chat.run_store import shared_chat_run_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine


class ActiveChatRun:
    """当前进程持有的对话生成句柄"""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self.cancellation_token = CancellationToken()
        self.task: asyncio.Task[None] | None = None
        self._activation = asyncio.Event()
        self._subscribers: set[asyncio.Queue[str | object]] = set()
        self._subscriber_end = object()
        self._closed = False

    @property
    def is_activated(self) -> bool:
        """任务是否已允许执行"""
        return self._activation.is_set()

    @property
    def subscriber_count(self) -> int:
        """当前本地订阅者数量"""
        return len(self._subscribers)

    def start(self, coroutine: Coroutine[Any, Any, None], *, conversation_id: str) -> None:
        """
        幂等启动后台生成任务

        :param coroutine: 后台生成协程
        :param conversation_id: 对话 ID
        :return:
        """
        if self.task is not None:
            coroutine.close()
            return
        self.task = asyncio.create_task(
            self._run_after_activation(coroutine),
            name=f'ai-chat-run-{conversation_id}',
        )

    def activate(self) -> None:
        """在业务事务提交后允许后台生成开始"""
        self._activation.set()

    def subscribe(self, *, max_events: int = 512) -> AsyncIterator[str]:
        """
        订阅当前进程产生的新事件

        :param max_events: 单个订阅者最大待发送事件数
        :return:
        """
        queue: asyncio.Queue[str | object] = asyncio.Queue(maxsize=max_events)
        if self._closed:
            queue.put_nowait(self._subscriber_end)
        else:
            self._subscribers.add(queue)
        return self._iter_subscriber(queue)

    def publish(self, event: str) -> None:
        """
        向当前进程的订阅者推送事件

        :param event: 已编码协议事件
        :return:
        """
        for queue in tuple(self._subscribers):
            if queue.full():
                self._close_subscriber(queue)
            else:
                queue.put_nowait(event)

    def close_subscribers(self) -> None:
        """结束全部本地事件订阅"""
        self._closed = True
        for queue in tuple(self._subscribers):
            self._subscribers.discard(queue)
            if not queue.full():
                queue.put_nowait(self._subscriber_end)

    async def cancel_before_activation(self) -> None:
        """取消尚未越过事务提交屏障的任务"""
        if self.task is None:
            return
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    async def _run_after_activation(self, coroutine: Coroutine[Any, Any, None]) -> None:
        try:
            await self._activation.wait()
        except asyncio.CancelledError:
            coroutine.close()
            raise
        await coroutine

    async def _iter_subscriber(self, queue: asyncio.Queue[str | object]) -> AsyncIterator[str]:
        try:
            while True:
                event = await queue.get()
                if event is self._subscriber_end:
                    break
                assert isinstance(event, str)
                yield event
                if self._closed and queue.empty():
                    break
        finally:
            self._subscribers.discard(queue)

    def _close_subscriber(self, queue: asyncio.Queue[str | object]) -> None:
        self._subscribers.discard(queue)
        with suppress(asyncio.QueueEmpty):
            while True:
                queue.get_nowait()
        queue.put_nowait(self._subscriber_end)


_runs: dict[str, ActiveChatRun] = {}


def get_local_run(conversation_id: str) -> ActiveChatRun | None:
    """
    获取当前进程持有的生成句柄

    :param conversation_id: 对话 ID
    :return:
    """
    run = _runs.get(conversation_id)
    if run is not None and run.task is not None and run.task.done():
        _runs.pop(conversation_id, None)
        return None
    return run


async def register_run(*, conversation_id: str, run_id: str) -> ActiveChatRun:
    """
    注册当前进程持有的生成句柄

    :param conversation_id: 对话 ID
    :param run_id: 运行 ID
    :return:
    """
    existing = get_local_run(conversation_id)
    if existing is not None:
        if await shared_chat_run_store.is_active(conversation_id=conversation_id):
            raise errors.ConflictError(msg='当前对话正在生成，请稍后再试')
        if existing.is_activated:
            existing.cancellation_token.cancel()
        else:
            existing.close_subscribers()
            await existing.cancel_before_activation()
        _runs.pop(conversation_id, None)
    run = ActiveChatRun(run_id=run_id)
    _runs[conversation_id] = run
    return run


def discard_run(conversation_id: str, run: ActiveChatRun) -> None:
    """
    移除当前进程的生成句柄

    :param conversation_id: 对话 ID
    :param run: 待移除的生成句柄
    :return:
    """
    if _runs.get(conversation_id) is run:
        _runs.pop(conversation_id, None)


async def has_active_run(conversation_id: str) -> bool:
    """
    根据共享租约判断对话是否正在生成

    :param conversation_id: 对话 ID
    :return:
    """
    return await shared_chat_run_store.is_active(conversation_id=conversation_id)


async def request_stop_run(conversation_id: str) -> bool:
    """
    请求当前执行者停止对话生成

    :param conversation_id: 对话 ID
    :return:
    """
    run = get_local_run(conversation_id)
    local_requested = run is not None
    if run is not None:
        run.cancellation_token.cancel()
    try:
        shared_requested = await shared_chat_run_store.request_cancel(conversation_id=conversation_id)
    except Exception:
        if local_requested:
            return True
        raise
    return shared_requested or local_requested


async def wait_run_stopped(
    conversation_id: str,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    """
    等待对话生成租约释放

    :param conversation_id: 对话 ID
    :param timeout: 最大等待秒数
    :param poll_interval: 轮询间隔秒数
    :return:
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while await has_active_run(conversation_id):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(poll_interval, remaining))
    return True


def activate_run(conversation_id: str) -> None:
    """
    在业务事务提交后启动已准备的生成任务

    :param conversation_id: 对话 ID
    :return:
    """
    run = get_local_run(conversation_id)
    if run is None:
        raise RuntimeError('聊天生成任务不存在')
    run.activate()


async def abort_prepared_run(conversation_id: str) -> bool:
    """
    释放当前进程中尚未启动的生成任务

    :param conversation_id: 对话 ID
    :return:
    """
    run = get_local_run(conversation_id)
    if run is None or run.is_activated:
        return False
    discard_run(conversation_id, run)
    run.close_subscribers()
    await run.cancel_before_activation()
    return await shared_chat_run_store.release(conversation_id=conversation_id, run_id=run.run_id)


async def shutdown_runs(*, timeout: float = 10.0) -> None:
    """
    停止本进程全部聊天任务并等待终态落库

    :param timeout: 优雅停止最大等待秒数
    :return:
    """
    runs = list(_runs.items())
    if not runs:
        return
    active_tasks: list[asyncio.Task[None]] = []
    for conversation_id, run in runs:
        if run.is_activated:
            run.cancellation_token.cancel()
            if run.task is not None:
                active_tasks.append(run.task)
        else:
            run.close_subscribers()
            await run.cancel_before_activation()
            try:
                await shared_chat_run_store.release(conversation_id=conversation_id, run_id=run.run_id)
            except Exception as exc:
                log.warning(f'关闭聊天任务时释放租约失败 conversation_id={conversation_id}: {exc}')
            discard_run(conversation_id, run)
    if not active_tasks:
        return
    _, pending = await asyncio.wait(active_tasks, timeout=timeout)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
