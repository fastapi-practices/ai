from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.parse import urlsplit

import httpx

from pydantic_ai.models import Model

from backend.plugin.ai.enums import AIProviderType


class ProviderAdapter(ABC):
    """供应商适配器基类"""

    provider_type: ClassVar[AIProviderType]

    @abstractmethod
    def create_model(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> Model:
        """
        创建供应商模型实例

        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :param http_client: 共享 HTTP 客户端
        :return:
        """

    async def aclose(self, model: Model) -> None:
        """
        关闭供应商私有客户端，默认无需关闭

        :param model: 模型实例
        :return:
        """
        return

    def validate_model_id(self, model_id: str) -> None:
        """
        校验模型 ID 格式，默认无需校验

        :param model_id: 模型 ID
        :return:
        """
        return

def normalize_provider_api_host(provider_type: int | AIProviderType, api_host: str) -> str:
    """
    标准化供应商 API 地址

    :param provider_type: 供应商类型
    :param api_host: API 地址
    :return:
    """
    api_host = api_host.strip().rstrip('/')
    if urlsplit(api_host).path.strip('/'):
        return api_host
    return api_host + AIProviderType(provider_type).default_api_path
