from backend.common.enums import IntEnum


class AIProviderType(IntEnum):
    """AI 供应商类型"""

    openai = 0
    anthropic = 1
    gemini = 2
    bedrock = 3
    groq = 4
    mistral = 5
    openrouter = 6
