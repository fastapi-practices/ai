from backend.common.enums import IntEnum, StrEnum


class AIDefaultModelScene(StrEnum):
    """AI 默认模型场景"""

    assistant = 'assistant'


class AIProviderType(IntEnum):
    """AI 供应商类型"""

    openai = 0
    anthropic = 1
    google = 2
    xai = 3
    openrouter = 4
    openai_responses = 5

    @property
    def default_api_path(self) -> str:
        return {
            AIProviderType.openai: '/v1',
            AIProviderType.anthropic: '',
            AIProviderType.google: '/v1beta',
            AIProviderType.xai: '/v1',
            AIProviderType.openrouter: '/api/v1',
            AIProviderType.openai_responses: '/v1',
        }[self]


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


class AIMessageStatus(StrEnum):
    """AI 消息状态"""

    pending = 'pending'
    success = 'success'
    error = 'error'
    interrupted = 'interrupted'
