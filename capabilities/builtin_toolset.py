from backend.plugin.ai.capabilities.base import function_tools_allowed
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.tools.chat_builtin_toolset import build_chat_builtin_capability


async def build_builtin_toolset_capability(ctx: CapabilityContext) -> CapabilityResult:  # ruff:ignore[unused-async]
    """
    构建项目内置 toolset 能力

    :param ctx: 能力构建上下文
    :return:
    """
    if not ctx.forwarded_props.enable_builtin_tools:
        return CapabilityResult(capability=None)
    if not function_tools_allowed(
        adapter=ctx.adapter,
        supports_tools=ctx.supports_tools,
        has_builtin_tools=ctx.has_builtin_tools,
    ):
        return CapabilityResult(capability=None)
    return CapabilityResult(
        capability=build_chat_builtin_capability(),
        introduces_function_tool_source=True,
    )
