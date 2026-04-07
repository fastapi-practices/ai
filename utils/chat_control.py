from typing import TypeAlias

from pydantic_ai import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings
from pydantic_ai.models.xai import XaiModelSettings

from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType, AIWebSearchType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

ChatModelSettings: TypeAlias = (
    ModelSettings
    | AnthropicModelSettings
    | GoogleModelSettings
    | OpenAIChatModelSettings
    | OpenAIResponsesModelSettings
    | OpenRouterModelSettings
    | XaiModelSettings
)


def build_model_settings(*, chat_metadata: AIChatForwardedPropsParam, provider_type: int) -> ChatModelSettings:  # noqa: C901
    """
    按供应商构建模型配置

    :param chat_metadata: 聊天元数据
    :param provider_type: 供应商类型
    :return:
    """
    provider = AIProviderType(provider_type)
    common_model_setting_fields = (
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
    )

    if provider == AIProviderType.anthropic:
        model_setting_fields = tuple(
            field
            for field in common_model_setting_fields
            if field not in {'seed', 'presence_penalty', 'frequency_penalty', 'logit_bias'}
        )
    elif provider == AIProviderType.google:
        model_setting_fields = tuple(
            field
            for field in common_model_setting_fields
            if field not in {'parallel_tool_calls', 'logit_bias', 'extra_body'}
        )
    elif provider == AIProviderType.xai:
        model_setting_fields = tuple(
            field
            for field in common_model_setting_fields
            if field not in {'timeout', 'seed', 'logit_bias', 'extra_headers', 'extra_body'}
        )
    else:
        model_setting_fields = common_model_setting_fields

    model_settings = chat_metadata.model_dump(include=set(model_setting_fields), exclude_unset=True, exclude_none=True)

    if provider == AIProviderType.openai:
        return OpenAIChatModelSettings(**model_settings)

    if provider == AIProviderType.openai_responses:
        if chat_metadata.generation_type == AIChatGenerationType.text and chat_metadata.enable_builtin_tools:
            model_settings['openai_include_code_execution_outputs'] = True
        if chat_metadata.web_search == AIWebSearchType.builtin:
            model_settings['openai_include_web_search_sources'] = True
        return OpenAIResponsesModelSettings(**model_settings)

    if provider == AIProviderType.anthropic:
        return AnthropicModelSettings(**model_settings)

    if provider == AIProviderType.google:
        return GoogleModelSettings(**model_settings)

    if provider == AIProviderType.xai:
        return XaiModelSettings(**model_settings)

    if provider == AIProviderType.openrouter:
        return OpenRouterModelSettings(**model_settings)

    return ModelSettings(**model_settings)
