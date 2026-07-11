import warnings

from collections.abc import Sequence

from pydantic_ai_harness.experimental import HarnessExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore', HarnessExperimentalWarning)
    from pydantic_ai_harness.experimental.compaction import ClampOversizedMessages, LimitWarner, SlidingWindow

from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatGenerationType


async def build_context_management_capabilities(  # noqa: RUF029
    ctx: CapabilityContext,
) -> Sequence[CapabilityResult]:
    """
    构建上下文管理能力

    :param ctx: 能力构建上下文
    :return:
    """
    if ctx.forwarded_props.generation_type != AIChatGenerationType.text:
        return ()

    policy = ctx.context_management
    results: list[CapabilityResult] = []
    if policy.max_part_chars is not None:
        results.append(
            CapabilityResult(
                capability=ClampOversizedMessages(max_part_chars=policy.max_part_chars),
            )
        )
    if policy.max_messages is not None:
        if policy.keep_messages >= policy.max_messages:
            raise ValueError('上下文保留消息数量必须小于最大消息数量')
        results.append(
            CapabilityResult(
                capability=SlidingWindow(
                    max_messages=policy.max_messages,
                    keep_messages=policy.keep_messages,
                ),
            )
        )
    if policy.max_tokens is not None:
        results.append(
            CapabilityResult(
                capability=LimitWarner(
                    max_context_tokens=policy.max_tokens,
                    warning_threshold=policy.warning_threshold,
                ),
            )
        )
    return tuple(results)
