from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.cohere import CohereProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.outlines import OutlinesProvider

from backend.common.exception import errors
from backend.plugin.ai.enums import AIProviderType


def get_pydantic_model(  # noqa: C901
    provider_type: int, model_name: str, api_key: str, base_url: str, model_settings: ModelSettings
) -> (
    OpenAIChatModel
    | AnthropicModel
    | GoogleModel
    | BedrockConverseModel
    | CerebrasModel
    | CohereModel
    | GroqModel
    | HuggingFaceModel
    | MistralModel
    | OpenRouterModel
    | OutlinesModel
):
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

    if provider_type == AIProviderType.openai:
        openai_base_url = None
        if base_url:
            openai_base_url = f'{base_url}/v1' if not base_url.endswith('/v1') else base_url
        return OpenAIChatModel(
            model_name, provider=OpenAIProvider(base_url=openai_base_url, api_key=api_key), settings=model_settings
        )

    if provider_type == AIProviderType.anthropic:
        anthropic_base_url = None
        if base_url:
            anthropic_base_url = f'{base_url}/v1' if not base_url.endswith('/v1') else base_url
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(base_url=anthropic_base_url, api_key=api_key),
            settings=model_settings,
        )

    if provider_type == AIProviderType.gemini:
        google_base_url = None
        if base_url:
            google_base_url = f'{base_url}/v1beta/openai' if not base_url.endswith('/v1beta/openai') else base_url
        return GoogleModel(
            model_name, provider=GoogleProvider(base_url=google_base_url, api_key=api_key), settings=model_settings
        )

    if provider_type == AIProviderType.bedrock:
        region = base_url if base_url and not base_url.startswith('http') else 'us-east-1'
        return BedrockConverseModel(
            model_name,  # type: ignore[arg-type]
            provider=BedrockProvider(api_key=api_key, region_name=region),
            settings=model_settings,
        )

    if provider_type == AIProviderType.cerebras:
        return CerebrasModel(model_name, provider=CerebrasProvider(api_key=api_key), settings=model_settings)  # type: ignore[arg-type]

    if provider_type == AIProviderType.cohere:
        return CohereModel(model_name, provider=CohereProvider(api_key=api_key), settings=model_settings)

    if provider_type == AIProviderType.groq:
        groq_base_url = None
        if base_url:
            groq_base_url = f'{base_url}/openai/v1' if not base_url.endswith('/openai/v1') else base_url
        return GroqModel(
            model_name, provider=GroqProvider(api_key=api_key, base_url=groq_base_url), settings=model_settings
        )

    if provider_type == AIProviderType.hugging_face:
        return HuggingFaceModel(
            model_name, provider=HuggingFaceProvider(base_url=base_url, api_key=api_key), settings=model_settings
        )

    if provider_type == AIProviderType.mistral:
        return MistralModel(model_name, provider=MistralProvider(api_key=api_key), settings=model_settings)

    if provider_type == AIProviderType.openrouter:
        openrouter_base_url = None
        if base_url:
            openrouter_base_url = f'{base_url}/api/v1' if not base_url.endswith('/api/v1') else base_url
        openai_client = AsyncOpenAI(base_url=openrouter_base_url, api_key=api_key) if openrouter_base_url else None
        return OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=api_key, openai_client=openai_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.outlines:
        return OutlinesModel(model_name, provider=OutlinesProvider(), settings=model_settings)  # type: ignore[arg-type]

    raise errors.NotFoundError(msg=f'不支持的供应商类型: {provider_type}')
