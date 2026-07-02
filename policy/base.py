from sqlalchemy.ext.asyncio import AsyncSession

from backend.plugin.ai.policy.context import AIInvocationContext, AIInvocationResult


class AIResourcePolicy:
    """AI 资源与调用策略基类"""

    async def before_invoke(self, *, db: AsyncSession, context: AIInvocationContext) -> None:
        """
        AI 调用前策略校验

        :param db: 数据库会话
        :param context: AI 调用策略上下文
        :return:
        """
        return

    async def after_invoke(
        self,
        *,
        db: AsyncSession,
        context: AIInvocationContext,
        result: AIInvocationResult,
    ) -> None:
        """
        AI 调用后策略通知

        :param db: 数据库会话
        :param context: AI 调用策略上下文
        :param result: AI 调用结果策略上下文
        :return:
        """
        return
