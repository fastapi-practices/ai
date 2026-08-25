import asyncio

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.common.log import log
from backend.plugin.ai.chat.runs import shutdown_runs
from backend.plugin.ai.service.conversation_service import ai_conversation_service

_RECONCILE_INTERVAL_SECONDS = 30


async def _reconcile_stale_pending_loop() -> None:
    while True:
        try:
            count = await ai_conversation_service.reconcile_stale_pending()
            if count:
                log.info(f'已收敛 {count} 条残留聊天消息')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(f'扫描残留聊天消息失败: {exc}')
        await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    管理聊天后台任务和残留状态收敛

    :param app: FastAPI 应用
    :return:
    """
    reconcile_task = asyncio.create_task(
        _reconcile_stale_pending_loop(),
        name='ai-chat-stale-pending-reconciler',
    )
    try:
        yield
    finally:
        reconcile_task.cancel()
        await asyncio.gather(reconcile_task, return_exceptions=True)
        await shutdown_runs()
