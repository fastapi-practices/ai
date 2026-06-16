from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias

from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter

CapabilityBuilder: TypeAlias = Callable[
    [CapabilityContext],
    Awaitable[CapabilityResult | Sequence[CapabilityResult]],
]


def function_tools_allowed(
    *,
    adapter: ProviderAdapter,
    supports_tools: bool,
    has_builtin_tools: bool,
) -> bool:
    """
    判断当前组合是否允许函数工具

    Google 不支持 builtin tools 与 function tools 混用

    :param adapter: 供应商适配器
    :param supports_tools: 模型是否支持 function tools
    :param has_builtin_tools: 是否已存在 builtin tools
    :return:
    """
    if not supports_tools:
        return False
    return not (adapter.provider_type == AIProviderType.google and has_builtin_tools)
