from backend.common.enums import IntEnum


class AIProviderType(IntEnum):
    """AI 供应商类型"""

    openai = 0
    anthropic = 1
    google = 2
    xai = 3
    openrouter = 4

    @property
    def default_api_path(self) -> str:
        return {
            AIProviderType.openai: '/v1',
            AIProviderType.anthropic: '',
            AIProviderType.google: '/v1beta',
            AIProviderType.xai: '/v1',
            AIProviderType.openrouter: '/api/v1',
        }[self]


class McpType(IntEnum):
    """Mcp 类型"""

    stdio = 0
    sse = 1
    streamable_http = 2
