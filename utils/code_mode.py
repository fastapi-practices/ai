from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness import CodeMode

from backend.plugin.ai.dataclasses import ChatAgentDeps
from backend.plugin.ai.enums import AIChatGenerationType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

CODE_MODE_TOOL_NAMES: tuple[str, ...] = (
    'get_current_time',
    'list_my_quick_phrases',
    'list_provider_models',
    'duckduckgo_search',
    'tavily_search',
    'exa_search',
    'exa_find_similar',
    'exa_get_contents',
    'exa_answer',
)


def build_code_mode_capability(
    *, forwarded_props: AIChatForwardedPropsParam
) -> AbstractCapability[ChatAgentDeps] | None:
    """
    按当前会话参数构建 CodeMode 能力

    :param forwarded_props: 聊天扩展参数
    :return:
    """
    if forwarded_props.generation_type != AIChatGenerationType.text:
        return None
    return CodeMode(tools=list(CODE_MODE_TOOL_NAMES))
