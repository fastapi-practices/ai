from inspect import isawaitable
from typing import ClassVar
from urllib.parse import urlsplit

import httpx

from pydantic_ai.models import Model
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
from pydantic_ai.providers.xai import XaiProvider
from xai_sdk import AsyncClient

from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter, ProviderCapabilities, normalize_provider_api_host


class XaiAdapter(ProviderAdapter):
    """xAI 供应商适配器

    使用 xai_sdk.AsyncClient 而非 httpx.AsyncClient，需要在 aclose 中显式关闭
    """

    provider_type: ClassVar[AIProviderType] = AIProviderType.xai
    settings_cls: ClassVar[type] = XaiModelSettings
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        excluded_setting_fields=frozenset({'timeout', 'seed', 'logit_bias', 'extra_headers', 'extra_body'}),
        supports_image_generation=False,
        image_supported_fields=None,
        extra_response_settings={},
    )

    def __init__(self) -> None:
        self._clients: dict[int, AsyncClient] = {}

    def create_model(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> Model:
        """
        创建 xAI 模型实例（http_client 入参未使用，xAI 走自有 gRPC 客户端）

        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :param http_client: 共享 HTTP 客户端
        :return:
        """
        base_url = normalize_provider_api_host(self.provider_type, base_url)
        parsed_url = urlsplit(base_url)
        xai_client = AsyncClient(
            api_key=api_key,
            api_host=parsed_url.netloc,
            use_insecure_channel=parsed_url.scheme == 'http',
        )
        model = XaiModel(model_name, provider=XaiProvider(xai_client=xai_client))
        self._clients[id(model)] = xai_client
        return model

    async def aclose(self, model: Model) -> None:
        """
        关闭 xAI 客户端

        :param model: 模型实例
        :return:
        """
        client = self._clients.pop(id(model), None)
        if client is None:
            return
        close = getattr(client, 'close', None)
        if close is None:
            return
        result = close()
        if isawaitable(result):
            await result
