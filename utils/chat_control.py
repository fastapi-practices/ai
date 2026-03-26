from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import ModelSettings, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings, OpenRouterReasoning
from pydantic_ai.models.xai import XaiModelSettings

from backend.common.exception import errors
from backend.plugin.ai.enums import AIChatOutputModeType, AIChatReasoningEffortType, AIProviderType
from backend.plugin.ai.schema.chat import AIChatParam

if TYPE_CHECKING:
    from anthropic.types.beta import BetaThinkingConfigEnabledParam
    from google.genai.types import ThinkingConfigDict
    from openai.types.shared.reasoning_effort import ReasoningEffort


class StructuredOutputBase(BaseModel):
    """结构化输出基础模型"""

    model_config = ConfigDict(extra='forbid')


def _build_openai_model_settings(*, chat: AIChatParam, common_settings: dict[str, Any]) -> OpenAIChatModelSettings:
    """
    构建 OpenAI 模型配置

    :param chat: 聊天参数
    :param common_settings: 通用模型配置
    :return:
    """

    openai_reasoning_effort: ReasoningEffort = None
    match chat.reasoning_effort:
        case AIChatReasoningEffortType.none:
            openai_reasoning_effort = 'none'
        case AIChatReasoningEffortType.minimal:
            openai_reasoning_effort = 'minimal'
        case AIChatReasoningEffortType.low:
            openai_reasoning_effort = 'low'
        case AIChatReasoningEffortType.medium:
            openai_reasoning_effort = 'medium'
        case AIChatReasoningEffortType.high:
            openai_reasoning_effort = 'high'
        case AIChatReasoningEffortType.xhigh:
            openai_reasoning_effort = 'xhigh'

    openai_settings = dict(common_settings)
    if openai_reasoning_effort is not None:
        openai_settings['openai_reasoning_effort'] = openai_reasoning_effort

    return OpenAIChatModelSettings(**openai_settings)


def _build_anthropic_model_settings(*, chat: AIChatParam, common_settings: dict[str, Any]) -> AnthropicModelSettings:
    """
    构建 Anthropic 模型配置

    :param chat: 聊天参数
    :param common_settings: 通用模型配置
    :return:
    """

    anthropic_thinking: BetaThinkingConfigEnabledParam | None = None
    if chat.include_thinking:
        anthropic_thinking: BetaThinkingConfigEnabledParam = {
            'type': 'enabled',
            'budget_tokens': 2048,
        }

    anthropic_settings = {
        k: v
        for k, v in common_settings.items()
        if k not in {'seed', 'presence_penalty', 'frequency_penalty', 'logit_bias'}
    }

    if anthropic_thinking is not None:
        anthropic_settings['anthropic_thinking'] = anthropic_thinking

    return AnthropicModelSettings(**anthropic_settings)


def _build_google_model_settings(*, chat: AIChatParam, common_settings: dict[str, Any]) -> GoogleModelSettings:
    """
    构建 Google 模型配置

    :param chat: 聊天参数
    :param common_settings: 通用模型配置
    :return:
    """

    google_thinking_config: ThinkingConfigDict | None = None
    if chat.include_thinking:
        google_thinking_config: ThinkingConfigDict = {'include_thoughts': True}

    google_settings = {
        k: v for k, v in common_settings.items() if k not in {'parallel_tool_calls', 'logit_bias', 'extra_body'}
    }

    if google_thinking_config is not None:
        google_settings['google_thinking_config'] = google_thinking_config
    return GoogleModelSettings(**google_settings)


def _build_xai_model_settings(*, common_settings: dict[str, Any]) -> XaiModelSettings:
    """
    构建 xAI 模型配置

    :param common_settings: 通用模型配置
    :return:
    """

    return XaiModelSettings(**{
        k: v for k, v in common_settings.items() if k not in {'seed', 'logit_bias', 'extra_body'}
    })


def _build_openrouter_model_settings(*, chat: AIChatParam, common_settings: dict[str, Any]) -> OpenRouterModelSettings:
    """
    构建 OpenRouter 模型配置

    :param chat: 聊天参数
    :param common_settings: 通用模型配置
    :return:
    """

    openrouter_reasoning: OpenRouterReasoning | None = None
    if chat.include_thinking or chat.reasoning_effort:
        openrouter_reasoning: OpenRouterReasoning = {
            'enabled': chat.include_thinking,
        }
        if chat.reasoning_effort == AIChatReasoningEffortType.low:
            openrouter_reasoning['effort'] = 'low'
        elif chat.reasoning_effort == AIChatReasoningEffortType.medium:
            openrouter_reasoning['effort'] = 'medium'
        elif chat.reasoning_effort == AIChatReasoningEffortType.high:
            openrouter_reasoning['effort'] = 'high'

    openrouter_settings = dict(common_settings)
    if openrouter_reasoning is not None:
        openrouter_settings['openrouter_reasoning'] = openrouter_reasoning

    return OpenRouterModelSettings(**openrouter_settings)


def build_model_settings(*, chat: AIChatParam, provider_type: int) -> ModelSettings | Any:
    """
    构建模型配置

    :param chat: 聊天参数
    :param provider_type: 供应商类型
    :return:
    """

    provider = AIProviderType(provider_type)
    requested_fields = chat.model_fields_set
    common_settings = {
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
    common_settings = {k: v for k, v in common_settings.items() if k in requested_fields and v is not None}

    if provider == AIProviderType.openai:
        return _build_openai_model_settings(chat=chat, common_settings=common_settings)
    if provider == AIProviderType.anthropic:
        return _build_anthropic_model_settings(chat=chat, common_settings=common_settings)
    if provider == AIProviderType.google:
        return _build_google_model_settings(chat=chat, common_settings=common_settings)
    if provider == AIProviderType.xai:
        return _build_xai_model_settings(common_settings=common_settings)
    if provider == AIProviderType.openrouter:
        return _build_openrouter_model_settings(chat=chat, common_settings=common_settings)

    return ModelSettings(**common_settings)


def build_schema_type(schema: dict[str, Any], *, model_name: str) -> Any:  # noqa: C901
    schema_type = schema.get('type')
    if isinstance(schema_type, list):
        non_null_types = [item for item in schema_type if item != 'null']
        if len(non_null_types) == 1:
            return build_schema_type({**schema, 'type': non_null_types[0]}, model_name=model_name) | None
        return Any

    if schema_type == 'string':
        return str
    if schema_type == 'integer':
        return int
    if schema_type == 'number':
        return float
    if schema_type == 'boolean':
        return bool
    if schema_type == 'null':
        return None
    if schema_type == 'array':
        return list[Any]
    if schema_type == 'object' or 'properties' in schema:
        fields: dict[str, tuple[Any, Any]] = {}
        required = set(schema.get('required', []))
        for field_name, field_schema in schema.get('properties', {}).items():
            field_type = build_schema_type(field_schema, model_name=f'{model_name}{field_name.title()}')
            if field_name in required:
                fields[field_name] = (field_type, Field(description=field_schema.get('description', field_name)))
            else:
                fields[field_name] = (
                    field_type | None,
                    Field(default=None, description=field_schema.get('description', field_name)),
                )
        if not fields:
            return dict[str, Any]
        return create_model(model_name, __base__=StructuredOutputBase, **fields)

    if 'anyOf' in schema:
        variants = [build_schema_type(item, model_name=f'{model_name}Variant') for item in schema['anyOf']]
        if not variants:
            return Any
        variant_type = variants[0]
        for item in variants[1:]:
            variant_type = variant_type | item
        return variant_type

    return Any


def build_output_type(*, chat: AIChatParam) -> Any:
    if chat.output_mode == AIChatOutputModeType.text:
        return str
    if not chat.output_schema:
        raise errors.RequestError(msg='结构化输出模式必须提供 output_schema')

    schema_type = build_schema_type(
        chat.output_schema,
        model_name=chat.output_schema_name or 'ChatStructuredOutput',
    )

    if chat.output_mode == AIChatOutputModeType.tool:
        return ToolOutput(schema_type, name=chat.output_schema_name, description=chat.output_schema_description)
    if chat.output_mode == AIChatOutputModeType.native:
        return NativeOutput(schema_type, name=chat.output_schema_name)
    if chat.output_mode == AIChatOutputModeType.prompted:
        return PromptedOutput(schema_type, name=chat.output_schema_name, description=chat.output_schema_description)

    raise errors.RequestError(msg='不支持的输出模式')
