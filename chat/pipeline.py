from collections.abc import Sequence
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.native_tools import AbstractNativeTool
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.plugin.ai.capabilities.base import CapabilityBuilder
from backend.plugin.ai.capabilities.builtin_toolset import build_builtin_toolset_capability
from backend.plugin.ai.capabilities.code_execution import build_code_execution_capability
from backend.plugin.ai.capabilities.code_mode import build_code_mode_capability
from backend.plugin.ai.capabilities.image import build_image_generation_capability
from backend.plugin.ai.capabilities.mcp import build_mcp_capability
from backend.plugin.ai.capabilities.search import build_search_capabilities
from backend.plugin.ai.capabilities.thinking import build_thinking_capability
from backend.plugin.ai.dataclasses import CapabilityContext
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

_CAPABILITY_BUILDERS: tuple[CapabilityBuilder, ...] = (
    build_thinking_capability,
    build_mcp_capability,
    build_search_capabilities,
    build_code_execution_capability,
    build_image_generation_capability,
    build_builtin_toolset_capability,
    build_code_mode_capability,
)


async def assemble_capabilities(
    *,
    db: AsyncSession,
    adapter: ProviderAdapter,
    forwarded_props: AIChatForwardedPropsParam,
    supports_tools: bool,
    supported_native_tools: frozenset[type[AbstractNativeTool]],
    supports_image_output: bool,
) -> list[AbstractCapability[Any]]:
    """
    按顺序运行 capability 构建器并校验能力组合

    :param db: 数据库会话
    :param adapter: 供应商适配器
    :param forwarded_props: 聊天扩展参数
    :param supports_tools: 是否支持函数工具
    :param supported_native_tools: 支持的原生工具集合
    :param supports_image_output: 是否支持图片输出
    :return:
    """
    capabilities: list[AbstractCapability[Any]] = []
    has_builtin = False
    has_fn_source = False

    for builder in _CAPABILITY_BUILDERS:
        ctx = CapabilityContext(
            db=db,
            adapter=adapter,
            forwarded_props=forwarded_props,
            supports_tools=supports_tools,
            supported_native_tools=supported_native_tools,
            supports_image_output=supports_image_output,
            has_builtin_tools=has_builtin,
            has_function_tool_sources=has_fn_source,
        )
        outcomes = await builder(ctx)
        normalized: Sequence[Any] = outcomes if isinstance(outcomes, Sequence) else (outcomes,)
        for result in normalized:
            if result.capability is not None:
                capabilities.append(result.capability)
            has_builtin = has_builtin or result.introduces_builtin_tool
            has_fn_source = has_fn_source or result.introduces_function_tool_source

    if has_fn_source and not supports_tools:
        raise errors.RequestError(msg='当前模型不支持函数工具，请关闭 MCP、本地搜索或项目内置工具')
    if has_fn_source and adapter.provider_type == AIProviderType.google and has_builtin:
        raise errors.RequestError(msg='Google 模型不支持同时使用内置工具和函数工具，请关闭 MCP 和本地搜索/关闭内置工具')
    return capabilities
