from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness import CodeMode

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


def should_enable_function_tools(
    *,
    provider_type: int,
    supports_tools: bool,
    has_builtin_tools: bool,
) -> bool:
    """
    判断是否允许函数工具

    Google 当前不支持将 builtin tools 与 function tools 混用

    :param provider_type: 供应商类型
    :param supports_tools: 模型是否支持 function tools
    :param has_builtin_tools: 当前能力列表是否包含 builtin tools
    :return:
    """
    if not supports_tools:
        return False
    return not (AIProviderType(provider_type) == AIProviderType.google and has_builtin_tools)


def build_code_mode_capability(
    *,
    forwarded_props: AIChatForwardedPropsParam,
    provider_type: int,
    supports_tools: bool,
    has_builtin_tools: bool,
) -> AbstractCapability[Any] | None:
    """
    按当前会话参数构建 CodeMode 能力

    :param forwarded_props: 聊天扩展参数
    :param provider_type: 供应商类型
    :param supports_tools: 模型是否支持 function tools
    :param has_builtin_tools: 当前能力列表是否包含 builtin tools
    :return:
    """
    if forwarded_props.generation_type != AIChatGenerationType.text:
        return None
    if not should_enable_function_tools(
        provider_type=provider_type,
        supports_tools=supports_tools,
        has_builtin_tools=has_builtin_tools,
    ):
        return None
    code_mode_tools = [tool.strip() for tool in settings.AI_CODE_MODE_TOOLS if tool.strip()]
    if not code_mode_tools:
        return None
    if 'all' in code_mode_tools:
        if len(code_mode_tools) > 1:
            raise errors.ServerError(msg='AI_CODE_MODE_TOOLS 配置 all 时不能混用其他工具名称')
        return CodeMode(tools='all')
    return CodeMode(tools=code_mode_tools)
