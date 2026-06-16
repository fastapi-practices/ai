from pydantic_ai.capabilities import Thinking

from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatThinkingType


async def build_thinking_capability(ctx: CapabilityContext) -> CapabilityResult:  # noqa: RUF029
    """
    构建 Thinking 能力

    :param ctx: 能力构建上下文
    :return:
    """
    if ctx.forwarded_props.thinking is None:
        return CapabilityResult(capability=None)
    thinking_effort = (
        ctx.forwarded_props.thinking.value
        if isinstance(ctx.forwarded_props.thinking, AIChatThinkingType)
        else ctx.forwarded_props.thinking
    )
    return CapabilityResult(capability=Thinking(thinking_effort))
