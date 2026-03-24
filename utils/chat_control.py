from pydantic_ai import Agent, ModelSettings

from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.schema.chat import AIChatParam

chat_agent = Agent(name='fba_chat')


def build_model_settings(*, chat: AIChatParam, provider_type: int) -> ModelSettings:
    raw_settings = {
        'max_tokens': chat.max_tokens,
        'temperature': chat.temperature,
        'top_p': chat.top_p,
        'timeout': chat.timeout,
        'parallel_tool_calls': chat.parallel_tool_calls,
        'seed': chat.seed,
        'presence_penalty': chat.presence_penalty,
        'frequency_penalty': chat.frequency_penalty,
        'logit_bias': chat.logit_bias,
        'stop_sequences': chat.stop_sequences,
        'extra_headers': chat.extra_headers,
        'extra_body': chat.extra_body,
    }

    supported_keys = {
        AIProviderType.openai: frozenset({
            'max_tokens',
            'temperature',
            'top_p',
            'timeout',
            'parallel_tool_calls',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'stop_sequences',
            'extra_headers',
            'extra_body',
        }),
        AIProviderType.anthropic: frozenset({
            'max_tokens',
            'temperature',
            'top_p',
            'timeout',
            'parallel_tool_calls',
            'stop_sequences',
            'extra_headers',
            'extra_body',
        }),
        AIProviderType.google: frozenset({
            'max_tokens',
            'temperature',
            'top_p',
            'timeout',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'stop_sequences',
            'extra_headers',
        }),
        AIProviderType.xai: frozenset({
            'max_tokens',
            'temperature',
            'top_p',
            'timeout',
            'parallel_tool_calls',
            'presence_penalty',
            'frequency_penalty',
            'stop_sequences',
            'extra_headers',
        }),
        AIProviderType.openrouter: frozenset({
            'max_tokens',
            'temperature',
            'top_p',
            'timeout',
            'parallel_tool_calls',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'stop_sequences',
            'extra_headers',
            'extra_body',
        }),
    }.get(AIProviderType(provider_type), frozenset())

    return ModelSettings(**{k: v for k, v in raw_settings.items() if k in supported_keys and v is not None})
