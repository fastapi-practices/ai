from backend.common.exception import errors
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.anthropic import AnthropicAdapter
from backend.plugin.ai.providers.base import ProviderAdapter
from backend.plugin.ai.providers.google import GoogleAdapter
from backend.plugin.ai.providers.openai import OpenAIAdapter
from backend.plugin.ai.providers.openai_responses import OpenAIResponsesAdapter
from backend.plugin.ai.providers.openrouter import OpenRouterAdapter
from backend.plugin.ai.providers.xai import XaiAdapter

_REGISTRY: dict[AIProviderType, ProviderAdapter] = {
    AIProviderType.openai: OpenAIAdapter(),
    AIProviderType.openai_responses: OpenAIResponsesAdapter(),
    AIProviderType.openrouter: OpenRouterAdapter(),
    AIProviderType.anthropic: AnthropicAdapter(),
    AIProviderType.google: GoogleAdapter(),
    AIProviderType.xai: XaiAdapter(),
}


def get_provider_adapter(provider_type: int | AIProviderType) -> ProviderAdapter:
    """
    获取供应商适配器

    :param provider_type: 供应商类型
    :return:
    """
    adapter = _REGISTRY.get(AIProviderType(provider_type))
    if adapter is None:
        raise errors.NotFoundError(msg=f'当前不支持此供应商: {provider_type}')
    return adapter
