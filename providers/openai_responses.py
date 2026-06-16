from typing import Any, ClassVar

import httpx

from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType, AIWebSearchType
from backend.plugin.ai.providers.base import ProviderAdapter, ProviderCapabilities, normalize_provider_api_host
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


class OpenAIResponsesAdapter(ProviderAdapter):
    """OpenAI Responses 供应商适配器"""

    provider_type: ClassVar[AIProviderType] = AIProviderType.openai_responses
    settings_cls: ClassVar[type] = OpenAIResponsesModelSettings
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        excluded_setting_fields=frozenset(),
        supports_image_generation=True,
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
        创建 OpenAI Responses 模型实例

        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :param http_client: 共享 HTTP 客户端
        :return:
        """
        base_url = normalize_provider_api_host(self.provider_type, base_url)
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=openai_client))

    def resolve_extra_settings(self, *, forwarded_props: AIChatForwardedPropsParam) -> dict[str, Any]:
        """
        根据请求附加 Responses 专属 settings

        :param forwarded_props: 聊天扩展参数
        :return:
        """
        extras: dict[str, Any] = {}
        if forwarded_props.generation_type == AIChatGenerationType.text and forwarded_props.enable_builtin_tools:
            extras['openai_include_code_execution_outputs'] = True
        if forwarded_props.web_search == AIWebSearchType.builtin:
            extras['openai_include_web_search_sources'] = True
        return extras
