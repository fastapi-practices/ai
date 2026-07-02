import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.log import log
from backend.plugin.ai.policy.base import AIResourcePolicy
from backend.plugin.ai.policy.context import AIInvocationContext, AIInvocationResult

_ai_resource_policies: list[AIResourcePolicy] = []


def register_ai_resource_policy(policy: AIResourcePolicy) -> None:
    """
    注册 AI 资源与调用策略

    :param policy: AI 资源与调用策略
    :return:
    """
    if policy not in _ai_resource_policies:
        _ai_resource_policies.append(policy)


async def validate_ai_invocation(*, db: AsyncSession, context: AIInvocationContext) -> None:
    """
    校验 AI 调用策略

    :param db: 数据库会话
    :param context: AI 调用策略上下文
    :return:
    """
    for policy in tuple(_ai_resource_policies):
        await policy.before_invoke(db=db, context=context)


async def notify_ai_invocation_result(
    *,
    db: AsyncSession,
    context: AIInvocationContext,
    result: AIInvocationResult,
) -> None:
    """
    通知 AI 调用结果

    :param db: 数据库会话
    :param context: AI 调用策略上下文
    :param result: AI 调用结果策略上下文
    :return:
    """
    results = await asyncio.gather(
        *[policy.after_invoke(db=db, context=context, result=result) for policy in tuple(_ai_resource_policies)],
        return_exceptions=True,
    )
    for item in results:
        if isinstance(item, Exception):
            log.warning(f'AI 调用后策略执行失败: {item}')
