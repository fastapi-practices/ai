from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from backend.common.exception import errors
from backend.plugin.ai.enums import AIProviderType


def get_pydantic_model(
    provider_type: int, model_name: str, api_key: str, base_url: str, model_settings: ModelSettings
) -> OpenAIChatModel | AnthropicModel | GoogleModel | BedrockConverseModel | GroqModel | MistralModel | OpenRouterModel:
    """
    获取 pydantic 模型

    :param provider_type: 供应商类型
    :param model_name: 模型名称
    :param api_key: 密钥
    :param base_url: API 基础域名
    :param model_settings: 模型配置
    :return:
    """
    base_url = base_url.rstrip('/') if base_url else None

    # OpenAI: https://api.openai.com/v1
    if provider_type == AIProviderType.openai:
        return OpenAIChatModel(
            model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key), settings=model_settings
        )

    # Anthropic: https://api.anthropic.com (SDK 内部自动添加 /v1)
    if provider_type == AIProviderType.anthropic:
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(base_url=base_url, api_key=api_key),
            settings=model_settings,
        )

    # Gemini: https://generativelanguage.googleapis.com
    if provider_type == AIProviderType.gemini:
        return GoogleModel(
            model_name, provider=GoogleProvider(base_url=base_url, api_key=api_key), settings=model_settings
        )

    # Bedrock: 使用 region_name，如 us-east-1
    if provider_type == AIProviderType.bedrock:
        region = base_url if base_url and not base_url.startswith('http') else 'us-east-1'
        return BedrockConverseModel(
            model_name,  # type: ignore[arg-type]
            provider=BedrockProvider(api_key=api_key, region_name=region),
            settings=model_settings,
        )

    # Groq: https://api.groq.com/openai/v1
    if provider_type == AIProviderType.groq:
        return GroqModel(model_name, provider=GroqProvider(api_key=api_key, base_url=base_url), settings=model_settings)

    # Mistral: https://api.mistral.ai/v1
    if provider_type == AIProviderType.mistral:
        return MistralModel(
            model_name, provider=MistralProvider(api_key=api_key, base_url=base_url), settings=model_settings
        )

    # OpenRouter: https://openrouter.ai/api/v1
    if provider_type == AIProviderType.openrouter:
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key) if base_url else None
        return OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=api_key, openai_client=openai_client),
            settings=model_settings,
        )

    raise errors.NotFoundError(msg=f'不支持的供应商类型: {provider_type}')
