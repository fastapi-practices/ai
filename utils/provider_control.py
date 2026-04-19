from urllib.parse import urlsplit

from backend.plugin.ai.enums import AIProviderType


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
