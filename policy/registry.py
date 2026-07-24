from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.log import log
from backend.plugin.ai.policy.base import AIResourcePolicy
from backend.plugin.ai.policy.context import AIInvocationContext, AIInvocationResult
from backend.plugin.ai.policy.runtime import begin_ai_policy_shared, end_ai_policy_shared

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

    按注册顺序串行执行：任一策略拒绝即中断。
    同一调用周期内通过共享缓存避免策略间重复查库。

    :param db: 数据库会话
    :param context: AI 调用策略上下文
    :return:
    """
    policies = tuple(_ai_resource_policies)
    if not policies:
        return

    # AsyncSession 不可并发，策略必须串行；共享缓存供策略复用查询结果
    _, token = begin_ai_policy_shared()
    try:
        for policy in policies:
            await policy.before_invoke(db=db, context=context)
    finally:
        end_ai_policy_shared(token)


async def notify_ai_invocation_result(
    *,
    db: AsyncSession,
    context: AIInvocationContext,
    result: AIInvocationResult,
) -> None:
    """
    通知 AI 调用结果

    按注册顺序串行通知；每个策略使用独立 savepoint，互不影响。
    同一调用周期内通过共享缓存避免策略间重复查库。

    :param db: 数据库会话
    :param context: AI 调用策略上下文
    :param result: AI 调用结果策略上下文
    :return:
    """
    policies = tuple(_ai_resource_policies)
    if not policies:
        return

    # AsyncSession 不可并发，策略必须串行；共享缓存供策略复用查询结果
    _, token = begin_ai_policy_shared()
    try:
        for policy in policies:
            try:
                # 单策略失败回滚自身变更，不影响其他策略与主流程
                async with db.begin_nested():
                    await policy.after_invoke(db=db, context=context, result=result)
            except Exception as exc:  # ruff:ignore[try-except-in-loop]
                log.warning(f'AI 调用后策略执行失败: {exc}')
    finally:
        end_ai_policy_shared(token)
