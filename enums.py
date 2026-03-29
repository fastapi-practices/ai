from backend.common.enums import IntEnum, StrEnum


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

    @property
    def supports_builtin_web_search(self) -> bool:
        """是否支持模型内置网络搜索"""

        return self in {
            AIProviderType.anthropic,
            AIProviderType.google,
            AIProviderType.xai,
            AIProviderType.openrouter,
        }


class McpType(IntEnum):
    """Mcp 类型"""

    stdio = 0
    sse = 1
    streamable_http = 2


class AIChatAttachmentType(StrEnum):
    """聊天附件类型"""

    image = 'image'
    audio = 'audio'
    video = 'video'
    document = 'document'


class AIChatAttachmentSourceType(StrEnum):
    """聊天附件来源类型"""

    url = 'url'
    base64 = 'base64'


class AIChatGenerationType(StrEnum):
    """聊天生成类型"""

    text = 'text'
    image = 'image'


class AIMessageRoleType(StrEnum):
    """消息角色"""

    user = 'user'
    thinking = 'thinking'
    model = 'model'


class AIChatOutputModeType(StrEnum):
    """聊天输出模式"""

    text = 'text'
    tool = 'tool'
    native = 'native'
    prompted = 'prompted'


class AIWebSearchType(StrEnum):
    """网络搜索模式"""

    builtin = 'builtin'
    tavily = 'tavily'
    duckduckgo = 'duckduckgo'


class AIChatReasoningEffortType(StrEnum):
    """聊天模型推理强度"""

    none = 'none'
    minimal = 'minimal'
    low = 'low'
    medium = 'medium'
    high = 'high'
    xhigh = 'xhigh'
