from urllib.parse import urlsplit

import httpx

from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt
from xai_sdk import AsyncClient

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.utils.provider_control import normalize_provider_api_host

ProviderModel = OpenAIChatModel | OpenAIResponsesModel | AnthropicModel | GoogleModel | XaiModel | OpenRouterModel


def _build_retry_http_client() -> httpx.AsyncClient:
    """
    构建带重试的 HTTP 客户端

    :return:
    """
    return httpx.AsyncClient(
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type((
                    httpx.HTTPStatusError,
                    httpx.TransportError,
                )),
                wait=wait_retry_after(),
                stop=stop_after_attempt(settings.AI_HTTP_MAX_RETRIES + 1),
                reraise=True,
            ),
            validate_response=lambda response: (
                response.raise_for_status() if response.status_code in {408, 409, 429, 500, 502, 503, 504} else None
            ),
        )
    )


def get_provider_model(
    provider_type: int,
    model_name: str,
    api_key: str,
    base_url: str,
    model_settings: ModelSettings,
) -> ProviderModel:
    """
    获取供应商模型

    :param provider_type: 供应商类型
    :param model_name: 模型名称
    :param api_key: 密钥
    :param base_url: API 基础域名
    :param model_settings: 模型配置
    :return:
    """
    provider_type = AIProviderType(provider_type)
    base_url = normalize_provider_api_host(provider_type, base_url)
    retry_http_client = _build_retry_http_client()

    if provider_type == AIProviderType.openai:
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=retry_http_client)
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(openai_client=openai_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.openai_responses:
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=retry_http_client)
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(openai_client=openai_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.openrouter:
        provider = (
            OpenRouterProvider(
                openai_client=AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=retry_http_client)
            )
            if base_url
            else OpenRouterProvider(api_key=api_key, http_client=retry_http_client)
        )
        return OpenRouterModel(
            model_name,
            provider=provider,
            settings=model_settings,
        )

    if provider_type == AIProviderType.anthropic:
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(base_url=base_url, api_key=api_key, http_client=retry_http_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.google:
        return GoogleModel(
            model_name,
            provider=GoogleProvider(base_url=base_url, api_key=api_key, http_client=retry_http_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.xai:
        parsed_url = urlsplit(base_url)
        return XaiModel(
            model_name,
            provider=XaiProvider(
                xai_client=AsyncClient(
                    api_key=api_key,
                    api_host=parsed_url.netloc,
                    use_insecure_channel=parsed_url.scheme == 'http',
                )
            ),
            settings=model_settings,
        )

    raise errors.NotFoundError(msg=f'当前不支持此供应商: {provider_type}')
