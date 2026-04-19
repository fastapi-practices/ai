from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness import CodeMode

from backend.plugin.ai.dataclasses import ChatAgentDeps
from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

CODE_MODE_TOOL_NAMES: tuple[str, ...] = ()


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


def should_enable_code_mode(
    *,
    forwarded_props: AIChatForwardedPropsParam,
    provider_type: int,
    supports_tools: bool,
    has_builtin_tools: bool,
) -> bool:
    """
    判断当前会话是否允许启用 CodeMode

    CodeMode 会向模型暴露 `run_code` 函数工具，因此需要同时满足文本生成与模型支持工具调用
    另外 Google 当前不支持将 builtin tools 与 function tools 混用，因此需要在存在 builtin tools 时禁用

    :param forwarded_props: 聊天扩展参数
    :param provider_type: 供应商类型
    :param supports_tools: 模型是否支持 function tools
    :param has_builtin_tools: 当前能力列表是否包含 builtin tools
    :return:
    """
    if forwarded_props.generation_type != AIChatGenerationType.text:
        return False
    return should_enable_function_tools(
        provider_type=provider_type,
        supports_tools=supports_tools,
        has_builtin_tools=has_builtin_tools,
    )


def build_code_mode_capability(
    *,
    forwarded_props: AIChatForwardedPropsParam,
    provider_type: int,
    supports_tools: bool,
    has_builtin_tools: bool,
) -> AbstractCapability[ChatAgentDeps] | None:
    """
    按当前会话参数构建 CodeMode 能力

    :param forwarded_props: 聊天扩展参数
    :param provider_type: 供应商类型
    :param supports_tools: 模型是否支持 function tools
    :param has_builtin_tools: 当前能力列表是否包含 builtin tools
    :return:
    """
    if not should_enable_code_mode(
        forwarded_props=forwarded_props,
        provider_type=provider_type,
        supports_tools=supports_tools,
        has_builtin_tools=has_builtin_tools,
    ):
        return None
    return CodeMode(tools=list(CODE_MODE_TOOL_NAMES))
