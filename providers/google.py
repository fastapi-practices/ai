import httpx

from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.providers.base import ProviderAdapter, normalize_provider_api_host


class GoogleAdapter(ProviderAdapter):
    """Google 供应商适配器"""

    provider_type = AIProviderType.google

    def create_model(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> Model:
        """
        创建 Google 模型实例

        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :param http_client: 共享 HTTP 客户端
        :return:
        """
        base_url = normalize_provider_api_host(self.provider_type, base_url)
        return GoogleModel(
            model_name,
            provider=GoogleProvider(base_url=base_url, api_key=api_key, http_client=http_client),
        )
