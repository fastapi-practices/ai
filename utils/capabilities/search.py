from typing import Any

from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebFetchTool, WebSearchTool
from pydantic_ai.capabilities import AbstractCapability, BuiltinTool, Toolset
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.toolsets import FunctionToolset

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.enums import AIWebSearchType


def build_search_capabilities(
    *,
    web_search: AIWebSearchType,
    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]],
    auto_web_fetch: bool = False,
) -> list[AbstractCapability[Any]]:
    """
    构建聊天搜索能力

    :param web_search: 网络搜索模式
    :param supported_builtin_tools: 模型支持的 builtin tool 类型
    :param auto_web_fetch: 是否在支持时自动启用 WebFetchTool
    :return:
    """
    capabilities: list[AbstractCapability[Any]] = []

    match web_search:
        case AIWebSearchType.off:
            return capabilities
        case AIWebSearchType.builtin:
            if WebSearchTool not in supported_builtin_tools:
                raise errors.RequestError(
                    msg='当前模型不支持内置联网搜索，请选择 Exa、Tavily、DuckDuckGo 或关闭联网搜索'
                )
            capabilities.append(BuiltinTool(WebSearchTool()))
        case AIWebSearchType.exa:
            if not settings.AI_EXA_API_KEY:
                raise errors.RequestError(msg='未配置 AI_EXA_API_KEY，无法启用 Exa 搜索')
            capabilities.append(Toolset(ExaToolset(api_key=settings.AI_EXA_API_KEY)))
        case AIWebSearchType.tavily:
            if not settings.AI_TAVILY_API_KEY:
                raise errors.RequestError(msg='未配置 AI_TAVILY_API_KEY，无法启用 Tavily 搜索')
            capabilities.append(Toolset(FunctionToolset([tavily_search_tool(api_key=settings.AI_TAVILY_API_KEY)])))
        case AIWebSearchType.duckduckgo:
            capabilities.append(Toolset(FunctionToolset([duckduckgo_search_tool()])))
        case _:
            raise errors.RequestError(msg='不支持的网络搜索模式')

    if auto_web_fetch and WebFetchTool in supported_builtin_tools:
        capabilities.append(BuiltinTool(WebFetchTool()))

    return capabilities
