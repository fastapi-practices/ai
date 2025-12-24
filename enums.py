from backend.common.enums import IntEnum


class AIProviderType(IntEnum):
    """AI 供应商类型"""

    openai = 0
    anthropic = 1
    gemini = 2
    bedrock = 3
    cerebras = 4
    cohere = 5
    groq = 6
    hugging_face = 7
    mistral = 8
    openrouter = 9
    outlines = 10
