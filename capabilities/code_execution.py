from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import CodeExecutionTool

from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatGenerationType


async def build_code_execution_capability(ctx: CapabilityContext) -> CapabilityResult:  # noqa: RUF029
    """
    构建代码执行能力

    :param ctx: 能力构建上下文
    :return:
    """
    if not ctx.forwarded_props.enable_builtin_tools:
        return CapabilityResult(capability=None)
    if ctx.forwarded_props.generation_type != AIChatGenerationType.text:
        return CapabilityResult(capability=None)
    if CodeExecutionTool not in ctx.supported_native_tools:
        return CapabilityResult(capability=None)
    return CapabilityResult(
        capability=NativeTool(CodeExecutionTool()),
        introduces_builtin_tool=True,
    )
