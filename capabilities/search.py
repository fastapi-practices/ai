from collections.abc import Sequence

from pydantic_ai.capabilities import NativeTool, Toolset
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool
from pydantic_ai.toolsets import FunctionToolset

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType, AIWebSearchType
from backend.utils.dynamic_config import load_ai_config


async def build_search_capabilities(ctx: CapabilityContext) -> Sequence[CapabilityResult]:
    """
    构建联网搜索能力

    :param ctx: 能力构建上下文
    :return:
    """
    forwarded_props = ctx.forwarded_props
    web_search = forwarded_props.web_search

    auto_web_fetch = (
        ctx.adapter.provider_type != AIProviderType.google
        and forwarded_props.enable_builtin_tools
        and forwarded_props.generation_type == AIChatGenerationType.text
    )
    auto_web_fetch_enabled = (
        auto_web_fetch and web_search != AIWebSearchType.off and WebFetchTool in ctx.supported_native_tools
    )

    results: list[CapabilityResult] = []

    match web_search:
        case AIWebSearchType.off:
            pass
        case AIWebSearchType.builtin:
            if WebSearchTool not in ctx.supported_native_tools:
                raise errors.RequestError(
                    msg='当前模型不支持内置联网搜索，请选择 Exa、Tavily、DuckDuckGo 或关闭联网搜索'
                )
            results.append(CapabilityResult(capability=NativeTool(WebSearchTool()), introduces_builtin_tool=True))
        case AIWebSearchType.exa:
            await load_ai_config(ctx.db)
            if not settings.AI_EXA_API_KEY:
                raise errors.RequestError(msg='未配置 AI_EXA_API_KEY，无法启用 Exa 搜索')
            results.append(
                CapabilityResult(
                    capability=Toolset(ExaToolset(api_key=settings.AI_EXA_API_KEY)),
                    introduces_function_tool_source=True,
                )
            )
        case AIWebSearchType.tavily:
            await load_ai_config(ctx.db)
            if not settings.AI_TAVILY_API_KEY:
                raise errors.RequestError(msg='未配置 AI_TAVILY_API_KEY，无法启用 Tavily 搜索')
            results.append(
                CapabilityResult(
                    capability=Toolset(FunctionToolset([tavily_search_tool(api_key=settings.AI_TAVILY_API_KEY)])),
                    introduces_function_tool_source=True,
                )
            )
        case AIWebSearchType.duckduckgo:
            results.append(
                CapabilityResult(
                    capability=Toolset(FunctionToolset([duckduckgo_search_tool()])),
                    introduces_function_tool_source=True,
                )
            )
        case _:
            raise errors.RequestError(msg='不支持的网络搜索模式')

    if auto_web_fetch_enabled:
        results.append(CapabilityResult(capability=NativeTool(WebFetchTool()), introduces_builtin_tool=True))

    return tuple(results)
