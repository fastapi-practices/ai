from importlib import import_module
from typing import cast

from backend.common.exception import errors
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter

_ADAPTER_PATHS: dict[AIProviderType, tuple[str, str]] = {
    AIProviderType.openai: ('backend.plugin.ai.providers.openai', 'OpenAIAdapter'),
    AIProviderType.openai_responses: ('backend.plugin.ai.providers.openai_responses', 'OpenAIResponsesAdapter'),
    AIProviderType.openrouter: ('backend.plugin.ai.providers.openrouter', 'OpenRouterAdapter'),
    AIProviderType.anthropic: ('backend.plugin.ai.providers.anthropic', 'AnthropicAdapter'),
    AIProviderType.google: ('backend.plugin.ai.providers.google', 'GoogleAdapter'),
    AIProviderType.xai: ('backend.plugin.ai.providers.xai', 'XaiAdapter'),
}
_REGISTRY: dict[AIProviderType, ProviderAdapter] = {}


def get_provider_adapter(provider_type: int | AIProviderType) -> ProviderAdapter:
    """
    获取供应商适配器

    :param provider_type: 供应商类型
    :return:
    """
    resolved_type = AIProviderType(provider_type)
    adapter = _REGISTRY.get(resolved_type)
    if adapter is not None:
        return adapter
    adapter_path = _ADAPTER_PATHS.get(resolved_type)
    if adapter_path is None:
        raise errors.NotFoundError(msg=f'当前不支持此供应商: {provider_type}')
    module_name, class_name = adapter_path
    adapter_cls = cast('type[ProviderAdapter]', getattr(import_module(module_name), class_name))
    adapter = adapter_cls()
    _REGISTRY[resolved_type] = adapter
    return adapter
