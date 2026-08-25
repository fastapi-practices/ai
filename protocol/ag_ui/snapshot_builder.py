from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, NamedTuple, TypeAlias, cast

from ag_ui.core import (
    ActivityMessage,
    AssistantMessage,
    DeveloperMessage,
    Message,
    ReasoningMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from pydantic_ai import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.ui.ag_ui import AGUIAdapter

from backend.plugin.ai.protocol.ag_ui.schema import (
    AIChatAgUiActivityMessageDetail,
    AIChatAgUiAssistantMessageDetail,
    AIChatAgUiDeveloperMessageDetail,
    AIChatAgUiMessagesSnapshotDetail,
    AIChatAgUiReasoningMessageDetail,
    AIChatAgUiSnapshotMessageDetail,
    AIChatAgUiSystemMessageDetail,
    AIChatAgUiToolMessageDetail,
    AIChatAgUiUserMessageDetail,
)

SnapshotMessage: TypeAlias = (
    AIChatAgUiUserMessageDetail
    | AIChatAgUiDeveloperMessageDetail
    | AIChatAgUiAssistantMessageDetail
    | AIChatAgUiSystemMessageDetail
    | AIChatAgUiToolMessageDetail
    | AIChatAgUiActivityMessageDetail
    | AIChatAgUiReasoningMessageDetail
)
SnapshotMetaValue: TypeAlias = datetime | str | int | None


class SnapshotMessageBuildConfig(NamedTuple):
    """快照消息构建配置"""

    message_type: type[Message]
    fragment_type: str
    detail_model: type[Any]
    extra_fields_getter: Callable[[Any], dict[str, Any]]
    primary_without_suffix: bool = False


_SNAPSHOT_MESSAGE_BUILD_CONFIGS: tuple[SnapshotMessageBuildConfig, ...] = (
    SnapshotMessageBuildConfig(
        UserMessage,
        'user',
        AIChatAgUiUserMessageDetail,
        lambda message: {'content': message.content},
    ),
    SnapshotMessageBuildConfig(
        DeveloperMessage,
        'developer',
        AIChatAgUiDeveloperMessageDetail,
        lambda message: {'content': message.content},
    ),
    SnapshotMessageBuildConfig(
        SystemMessage,
        'system',
        AIChatAgUiSystemMessageDetail,
        lambda message: {'content': message.content},
    ),
    SnapshotMessageBuildConfig(
        AssistantMessage,
        'assistant',
        AIChatAgUiAssistantMessageDetail,
        lambda message: {
            'content': message.content,
            'tool_calls': message.tool_calls,
            'encrypted_value': message.encrypted_value,
        },
        primary_without_suffix=True,
    ),
    SnapshotMessageBuildConfig(
        ReasoningMessage,
        'reasoning',
        AIChatAgUiReasoningMessageDetail,
        lambda message: {
            'content': message.content,
            'encrypted_value': message.encrypted_value,
        },
    ),
    SnapshotMessageBuildConfig(
        ToolMessage,
        'tool',
        AIChatAgUiToolMessageDetail,
        lambda message: {
            'content': message.content,
            'tool_call_id': message.tool_call_id,
            'error': getattr(message, 'error', None),
            # 透传 outcome 等非 success 状态（pydantic-ai 编码于 encrypted_value）
            'encrypted_value': message.encrypted_value,
        },
    ),
    SnapshotMessageBuildConfig(
        ActivityMessage,
        'activity',
        AIChatAgUiActivityMessageDetail,
        lambda message: {
            'activity_type': message.activity_type,
            'content': cast('dict[str, Any]', message.content),
        },
    ),
)


def _build_snapshot_messages_from_encoded_messages(
    *,
    encoded_messages: Sequence[Message],
    base_meta: dict[str, SnapshotMetaValue],
    message_id: int | None,
    message_index: int,
    fallback_empty_assistant: bool = False,
) -> list[SnapshotMessage]:
    """
    根据标准 AG-UI 消息构建快照消息

    :param encoded_messages: 标准 AG-UI 消息列表
    :param base_meta: 公共元信息
    :param message_id: 持久化消息 ID
    :param message_index: 消息索引
    :param fallback_empty_assistant: 是否在空结果时补一条空助手消息
    :return:
    """
    snapshot_messages: list[SnapshotMessage] = []
    fragment_indexes: defaultdict[str, int] = defaultdict(int)

    for encoded_message in encoded_messages:
        for config in _SNAPSHOT_MESSAGE_BUILD_CONFIGS:
            if not isinstance(encoded_message, config.message_type):
                continue
            fragment_index = fragment_indexes[config.fragment_type]
            base_id = f'msg_{message_id if message_id is not None else message_index}'
            primary_without_suffix = not snapshot_messages and config.primary_without_suffix and fragment_index == 0
            snapshot_id = base_id if primary_without_suffix else f'{base_id}_{config.fragment_type}_{fragment_index}'
            snapshot_messages.append(
                cast(
                    'SnapshotMessage',
                    config.detail_model(
                        id=snapshot_id,
                        **config.extra_fields_getter(encoded_message),
                        **base_meta,
                    ),
                )
            )
            fragment_indexes[config.fragment_type] += 1
            break
        else:
            raise ValueError(f'不支持的 AG-UI 消息类型: {type(encoded_message).__name__}')

    if snapshot_messages or not fallback_empty_assistant:
        return snapshot_messages
    return [
        AIChatAgUiAssistantMessageDetail(
            id=f'msg_{message_id if message_id is not None else message_index}',
            content=None,
            tool_calls=None,
            **base_meta,
        )
    ]


def serialize_request_message(
    *,
    message: ModelRequest,
    conversation_id: str | None,
    message_id: int | None,
    provider_id: int | None,
    model_id: str | None,
    message_index: int,
) -> list[SnapshotMessage]:
    """
    序列化请求消息

    :param message: 请求消息
    :param conversation_id: 对话 ID
    :param message_id: 持久化消息 ID
    :param provider_id: 供应商 ID
    :param model_id: 模型 ID
    :param message_index: 消息索引
    :return:
    """
    if not message.parts:
        return []

    base_meta = {
        'conversation_id': conversation_id,
        'persisted_message_id': message_id,
        'provider_id': provider_id,
        'model_id': model_id,
        'created_time': message.parts[0].timestamp,
        'message_index': message_index,
        'message_type': message.state if message.state != 'complete' else 'normal',
    }

    encoded_messages = AGUIAdapter.dump_messages([message], preserve_file_data=True)
    return _build_snapshot_messages_from_encoded_messages(
        encoded_messages=encoded_messages,
        base_meta=base_meta,
        message_id=message_id,
        message_index=message_index,
    )


def serialize_response_message(
    *,
    message: ModelResponse,
    conversation_id: str | None,
    message_id: int | None,
    provider_id: int | None,
    model_id: str | None,
    message_index: int,
) -> list[SnapshotMessage]:
    """
    序列化响应消息

    :param message: 响应消息
    :param conversation_id: 对话 ID
    :param message_id: 持久化消息 ID
    :param provider_id: 供应商 ID
    :param model_id: 模型 ID
    :param message_index: 消息索引
    :return:
    """
    base_meta = {
        'conversation_id': conversation_id,
        'persisted_message_id': message_id,
        'provider_id': provider_id,
        'model_id': model_id or message.model_name,
        'created_time': message.timestamp,
        'message_index': message_index,
        # 对齐 pydantic-ai ModelResponse.state（complete/incomplete/suspended/interrupted）
        'message_type': (
            'error'
            if (message.metadata or {}).get('is_error')
            else 'normal'
            if message.state == 'complete'
            else message.state
        ),
    }

    encoded_messages = AGUIAdapter.dump_messages([message], preserve_file_data=True)
    return _build_snapshot_messages_from_encoded_messages(
        encoded_messages=encoded_messages,
        base_meta=base_meta,
        message_id=message_id,
        message_index=message_index,
        fallback_empty_assistant=True,
    )


def serialize_messages_to_snapshot(
    messages: Sequence[ModelMessage],
    *,
    conversation_id: str | None = None,
    message_ids: Sequence[int | None] | None = None,
    provider_ids: Sequence[int | None] | None = None,
    model_ids: Sequence[str | None] | None = None,
    message_indexes: Sequence[int | None] | None = None,
) -> AIChatAgUiMessagesSnapshotDetail:
    """
    序列化模型消息为快照

    :param messages: 模型消息列表
    :param conversation_id: 对话 ID
    :param message_ids: 持久化消息 ID 列表
    :param provider_ids: 供应商 ID 列表
    :param model_ids: 模型 ID 列表
    :param message_indexes: 持久化消息索引列表
    :return:
    """
    snapshot_messages: list[AIChatAgUiSnapshotMessageDetail] = []
    message_contexts = zip(
        messages,
        message_ids or [None] * len(messages),
        provider_ids or [None] * len(messages),
        model_ids or [None] * len(messages),
        message_indexes or [None] * len(messages),
        strict=False,
    )
    for fallback_index, (message, message_id, provider_id, model_id, message_index) in enumerate(message_contexts):
        resolved_message_index = fallback_index if message_index is None else message_index
        if isinstance(message, ModelRequest):
            request_messages = serialize_request_message(
                message=message,
                conversation_id=conversation_id,
                message_id=message_id,
                provider_id=provider_id,
                model_id=model_id,
                message_index=resolved_message_index,
            )
            snapshot_messages.extend(request_messages)
            continue
        if isinstance(message, ModelResponse):
            response_messages = serialize_response_message(
                message=message,
                conversation_id=conversation_id,
                message_id=message_id,
                provider_id=provider_id,
                model_id=model_id,
                message_index=resolved_message_index,
            )
            snapshot_messages.extend(response_messages)

    return AIChatAgUiMessagesSnapshotDetail(messages=snapshot_messages)
