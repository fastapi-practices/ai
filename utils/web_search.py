from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebFetchTool, WebSearchTool
from pydantic_ai.capabilities import BuiltinTool, Toolset
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.common_tools.tavily import tavily_search_tool

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.enums import AIWebSearchType

ChatSearchCapability = BuiltinTool | Toolset


def build_chat_search_tools(
    *,
    web_search: AIWebSearchType,
    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]],
    auto_web_fetch: bool = False,
) -> tuple[list[object], list[ChatSearchCapability]]:
    """
    构建聊天搜索工具和能力

    :param web_search: 网络搜索模式
    :param supported_builtin_tools: 模型支持的 builtin tool 类型
    :param auto_web_fetch: 是否在支持时自动启用 WebFetchTool
    :return:
    """
    tools: list[object] = []
    capabilities: list[ChatSearchCapability] = []

    if auto_web_fetch and WebFetchTool in supported_builtin_tools:
        capabilities.append(BuiltinTool(WebFetchTool()))

    if web_search == AIWebSearchType.builtin:
        if WebSearchTool in supported_builtin_tools:
            capabilities.append(BuiltinTool(WebSearchTool()))
        return tools, capabilities

    if web_search == AIWebSearchType.exa:
        capabilities.append(
            Toolset(
                ExaToolset(
                    api_key=settings.AI_EXA_API_KEY,
                    num_results=settings.AI_EXA_NUM_RESULTS,
                    max_characters=settings.AI_EXA_MAX_CHARACTERS,
                )
            )
        )
        return tools, capabilities

    if web_search == AIWebSearchType.tavily:
        tools.append(tavily_search_tool(api_key=settings.AI_TAVILY_API_KEY))
        return tools, capabilities

    if web_search == AIWebSearchType.duckduckgo:
        tools.append(duckduckgo_search_tool())
        return tools, capabilities

    raise errors.RequestError(msg='不支持的网络搜索模式')
