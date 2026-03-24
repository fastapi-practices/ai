from inspect import signature
from typing import Any, cast, get_args
from urllib.parse import urlparse, urlsplit

from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models import OpenAIChatCompatibleProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.xai import XaiProvider

from backend.common.exception import errors
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.utils.provider_url import normalize_provider_api_host

from xai_sdk import AsyncClient

PydanticAIModel = OpenAIChatModel | AnthropicModel | GoogleModel | XaiModel | OpenRouterModel

OPENAI_COMPATIBLE_PROVIDER_NAMES = frozenset(get_args(OpenAIChatCompatibleProvider.__value__))
OPENAI_COMPATIBLE_PROVIDER_ALIASES: dict[str, frozenset[str]] = {
    'alibaba': frozenset({'alibaba', 'dashscope', 'qwen'}),
    'azure': frozenset({'azure', 'azureopenai'}),
    'cerebras': frozenset({'cerebras'}),
    'deepseek': frozenset({'deepseek'}),
    'fireworks': frozenset({'fireworks'}),
    'github': frozenset({'github', 'githubmodels'}),
    'grok': frozenset({'grok', 'xai'}),
    'heroku': frozenset({'heroku'}),
    'litellm': frozenset({'litellm', 'lite'}),
    'moonshotai': frozenset({'moonshotai', 'moonshot', 'kimi'}),
    'nebius': frozenset({'nebius'}),
    'ollama': frozenset({'ollama'}),
    'openrouter': frozenset({'openrouter'}),
    'ovhcloud': frozenset({'ovhcloud', 'ovh'}),
    'sambanova': frozenset({'sambanova'}),
    'together': frozenset({'together', 'togetherai'}),
    'vercel': frozenset({'vercel'}),
}
def get_pydantic_model(
    provider_type: int,
    model_name: str,
    api_key: str,
    base_url: str,
    model_settings: ModelSettings,
    provider_name: str | None = None,
) -> PydanticAIModel:
    """
    获取 pydantic 模型

    :param provider_type: 供应商类型
    :param model_name: 模型名称
    :param api_key: 密钥
    :param base_url: API 基础域名
    :param model_settings: 模型配置
    :param provider_name: 供应商名称，用于推断 OpenAI-compatible provider
    :return:
    """
    provider_type = AIProviderType(provider_type)
    base_url = normalize_provider_api_host(provider_type, base_url)
    if provider_type == AIProviderType.openai:
        parsed_url = urlparse(base_url)
        normalized_candidates = [
            normalized
            for candidate in (provider_name, parsed_url.hostname, parsed_url.netloc, base_url)
            if candidate and (normalized := ''.join(ch for ch in candidate.casefold() if ch.isalnum()))
        ]
        inferred_provider_name = next(
            (normalized for normalized in normalized_candidates if normalized in OPENAI_COMPATIBLE_PROVIDER_NAMES),
            None,
        )
        if inferred_provider_name is None:
            inferred_provider_name = next(
                (
                    compatible_provider_name
                    for normalized in normalized_candidates
                    for compatible_provider_name, aliases in OPENAI_COMPATIBLE_PROVIDER_ALIASES.items()
                    if compatible_provider_name in OPENAI_COMPATIBLE_PROVIDER_NAMES
                    if any(alias in normalized for alias in aliases)
                ),
                None,
            )

        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        provider = OpenAIProvider(openai_client=openai_client)
        if inferred_provider_name not in {None, 'azure'}:
            try:
                provider_class = cast(type[Any], infer_provider_class(inferred_provider_name))
                init_parameters = signature(provider_class.__init__).parameters
                provider_kwargs = {'api_key': api_key} if 'api_key' in init_parameters else {}
                if base_url:
                    provider_kwargs.update(
                        {'base_url': base_url}
                        if 'base_url' in init_parameters
                        else {'api_base': base_url}
                        if 'api_base' in init_parameters
                        else {'openai_client': openai_client}
                        if 'openai_client' in init_parameters
                        else {}
                    )
                provider = provider_class(**provider_kwargs)
            except (ImportError, TypeError, ValueError):
                pass

        return OpenAIChatModel(model_name, provider=provider, settings=model_settings)

    if provider_type == AIProviderType.openrouter:
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key) if base_url else None
        return OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=api_key, openai_client=openai_client),
            settings=model_settings,
        )

    if provider_type == AIProviderType.anthropic:
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(base_url=base_url, api_key=api_key),
            settings=model_settings,
        )

    if provider_type == AIProviderType.google:
        return GoogleModel(
            model_name,
            provider=GoogleProvider(base_url=base_url, api_key=api_key),
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
