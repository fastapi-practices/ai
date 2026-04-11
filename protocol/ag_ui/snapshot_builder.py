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


def _build_fragment_snapshot_id(
    *,
    message_id: int | None,
    message_index: int,
    fragment_type: str,
    fragment_index: int,
    primary_without_suffix: bool = False,
) -> str:
    """
    构建分片快照消息 ID

    :param message_id: 持久化消息 ID
    :param message_index: 消息索引
    :param fragment_type: 分片类型
    :param fragment_index: 分片索引
    :param primary_without_suffix: 首个分片是否省略后缀
    :return:
    """
    base_id = f'msg_{message_id if message_id is not None else message_index}'
    if primary_without_suffix and fragment_index == 0:
        return base_id
    return f'{base_id}_{fragment_type}_{fragment_index}'


def _build_snapshot_message(
    *,
    encoded_message: Message,
    base_meta: dict[str, SnapshotMetaValue],
    message_id: int | None,
    message_index: int,
    fragment_indexes: defaultdict[str, int],
    primary_without_suffix: bool,
) -> SnapshotMessage:
    """
    根据标准 AG-UI 消息构建单条快照消息

    :param encoded_message: 标准 AG-UI 消息
    :param base_meta: 公共元信息
    :param message_id: 持久化消息 ID
    :param message_index: 消息索引
    :param fragment_indexes: 各分片类型索引
    :param primary_without_suffix: 是否为首条输出消息
    :return:
    """
    for config in _SNAPSHOT_MESSAGE_BUILD_CONFIGS:
        if not isinstance(encoded_message, config.message_type):
            continue
        fragment_index = fragment_indexes[config.fragment_type]
        snapshot_message = config.detail_model(
            id=_build_fragment_snapshot_id(
                message_id=message_id,
                message_index=message_index,
                fragment_type=config.fragment_type,
                fragment_index=fragment_index,
                primary_without_suffix=primary_without_suffix and config.primary_without_suffix,
            ),
            **config.extra_fields_getter(encoded_message),
            **base_meta,
        )
        fragment_indexes[config.fragment_type] += 1
        return cast('SnapshotMessage', snapshot_message)
    raise ValueError(f'不支持的 AG-UI 消息类型: {type(encoded_message).__name__}')


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
        snapshot_messages.append(
            _build_snapshot_message(
                encoded_message=encoded_message,
                base_meta=base_meta,
                message_id=message_id,
                message_index=message_index,
                fragment_indexes=fragment_indexes,
                primary_without_suffix=not snapshot_messages,
            )
        )

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
        'message_type': 'normal',
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
        'message_type': 'error' if (message.metadata or {}).get('is_error') else 'normal',
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
) -> AIChatAgUiMessagesSnapshotDetail:
    """
    序列化模型消息为快照

    :param messages: 模型消息列表
    :param conversation_id: 对话 ID
    :param message_ids: 持久化消息 ID 列表
    :param provider_ids: 供应商 ID 列表
    :param model_ids: 模型 ID 列表
    :return:
    """
    snapshot_messages: list[AIChatAgUiSnapshotMessageDetail] = []
    message_contexts = zip(
        messages,
        message_ids or [None] * len(messages),
        provider_ids or [None] * len(messages),
        model_ids or [None] * len(messages),
        strict=False,
    )
    for message, message_id, provider_id, model_id in message_contexts:
        message_index = len(snapshot_messages)
        if isinstance(message, ModelRequest):
            snapshot_messages.extend(
                serialize_request_message(
                    message=message,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    provider_id=provider_id,
                    model_id=model_id,
                    message_index=message_index,
                )
            )
            continue
        if isinstance(message, ModelResponse):
            snapshot_messages.extend(
                serialize_response_message(
                    message=message,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    provider_id=provider_id,
                    model_id=model_id,
                    message_index=message_index,
                )
            )

    return AIChatAgUiMessagesSnapshotDetail(messages=snapshot_messages)
