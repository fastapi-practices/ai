from typing import ClassVar

import httpx

from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter, ProviderCapabilities, normalize_provider_api_host


class OpenAIAdapter(ProviderAdapter):
    """OpenAI Chat 供应商适配器"""

    provider_type: ClassVar[AIProviderType] = AIProviderType.openai
    settings_cls: ClassVar[type] = OpenAIChatModelSettings
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        excluded_setting_fields=frozenset(),
        supports_image_generation=False,
        image_supported_fields=None,
        extra_response_settings={},
    )

    def create_model(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> Model:
        """
        创建 OpenAI Chat 模型实例

        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :param http_client: 共享 HTTP 客户端
        :return:
        """
        base_url = normalize_provider_api_host(self.provider_type, base_url)
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=openai_client))
