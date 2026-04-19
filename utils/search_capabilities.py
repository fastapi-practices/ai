from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebFetchTool, WebSearchTool
from pydantic_ai.capabilities import AbstractCapability, BuiltinTool, Toolset, WebSearch
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.common_tools.tavily import tavily_search_tool

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.dataclasses import ChatAgentDeps
from backend.plugin.ai.enums import AIWebSearchType

ChatSearchCapability = AbstractCapability[ChatAgentDeps]


def build_search_capabilities(
    *,
    web_search: AIWebSearchType,
    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]],
    auto_web_fetch: bool = False,
) -> list[ChatSearchCapability]:
    """
    构建聊天搜索能力

    :param web_search: 网络搜索模式
    :param supported_builtin_tools: 模型支持的 builtin tool 类型
    :param auto_web_fetch: 是否在支持时自动启用 WebFetchTool
    :return:
    """
    capabilities: list[ChatSearchCapability] = []

    if auto_web_fetch and WebFetchTool in supported_builtin_tools:
        capabilities.append(BuiltinTool(WebFetchTool()))

    if web_search == AIWebSearchType.builtin:
        if WebSearchTool in supported_builtin_tools:
            capabilities.append(BuiltinTool(WebSearchTool()))
        return capabilities

    if web_search == AIWebSearchType.exa:
        if not settings.AI_EXA_API_KEY:
            raise errors.RequestError(msg='未配置 AI_EXA_API_KEY，无法启用 Exa 搜索')
        capabilities.append(Toolset(ExaToolset(api_key=settings.AI_EXA_API_KEY)))
        return capabilities

    if web_search == AIWebSearchType.tavily:
        if not settings.AI_TAVILY_API_KEY:
            raise errors.RequestError(msg='未配置 AI_TAVILY_API_KEY，无法启用 Tavily 搜索')
        capabilities.append(WebSearch(builtin=False, local=tavily_search_tool(api_key=settings.AI_TAVILY_API_KEY)))
        return capabilities

    if web_search == AIWebSearchType.duckduckgo:
        capabilities.append(WebSearch(builtin=False, local=duckduckgo_search_tool()))
        return capabilities

    raise errors.RequestError(msg='不支持的网络搜索模式')
