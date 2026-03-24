from urllib.parse import urlsplit

from backend.plugin.ai.enums import AIProviderType


def normalize_provider_api_host(provider_type: int | AIProviderType, api_host: str) -> str:
    api_host = api_host.strip().rstrip('/')
    if urlsplit(api_host).path.strip('/'):
        return api_host
    return api_host + AIProviderType(provider_type).default_api_path
