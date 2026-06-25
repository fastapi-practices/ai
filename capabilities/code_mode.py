from pydantic_ai_harness import CodeMode

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.capabilities.base import function_tools_allowed
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatGenerationType


async def build_code_mode_capability(ctx: CapabilityContext) -> CapabilityResult:  # noqa: RUF029
    """
    构建 CodeMode 能力

    :param ctx: 能力构建上下文
    :return:
    """
    if ctx.forwarded_props.generation_type != AIChatGenerationType.text:
        return CapabilityResult(capability=None)
    if not function_tools_allowed(
        adapter=ctx.adapter,
        supports_tools=ctx.supports_tools,
        has_builtin_tools=ctx.has_builtin_tools,
    ):
        return CapabilityResult(capability=None)

    code_mode_tools = [tool.strip() for tool in settings.AI_CODE_MODE_TOOLS if tool.strip()]
    if not code_mode_tools:
        return CapabilityResult(capability=None)

    if 'all' in code_mode_tools:
        if len(code_mode_tools) > 1:
            raise errors.ServerError(msg='AI_CODE_MODE_TOOLS 配置 all 时不能混用其他工具名称')
        return CapabilityResult(
            capability=CodeMode(
                tools='all',
                max_retries=settings.AI_CODE_MODE_MAX_RETRIES,
                dynamic_catalog=settings.AI_CODE_MODE_DYNAMIC_CATALOG,
            )
        )

    return CapabilityResult(
        capability=CodeMode(
            tools=code_mode_tools,
            max_retries=settings.AI_CODE_MODE_MAX_RETRIES,
            dynamic_catalog=settings.AI_CODE_MODE_DYNAMIC_CATALOG,
        )
    )
