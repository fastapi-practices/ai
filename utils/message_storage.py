from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, ModelResponse

from backend.plugin.ai.model import AIMessage

ChatMessageRole: TypeAlias = Literal['assistant', 'user']
StoredModelMessage: TypeAlias = ModelRequest | ModelResponse


def build_chat_message_record(
    *,
    role: ChatMessageRole,
    model_messages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    构建聊天消息记录字段

    :param role: 聊天消息角色
    :param model_messages: 原始 Pydantic 模型消息
    :return:
    """
    return {
        'role': role,
        'model_messages': list(model_messages),
    }


def get_message_row_model_message_payloads(row: AIMessage) -> list[dict[str, Any]]:
    """
    获取消息行中的原始模型消息列表

    :param row: 消息行
    :return:
    """
    model_messages = getattr(row, 'model_messages', None)
    if isinstance(model_messages, list):
        return model_messages
    return []


def expand_message_rows(
    message_rows: Sequence[AIMessage],
) -> tuple[list[StoredModelMessage], list[tuple[int, int]]]:
    """
    展开消息行中的原始模型消息

    :param message_rows: 消息行
    :return:
    """
    raw_messages: list[dict[str, Any]] = []
    row_message_ranges: list[tuple[int, int]] = []
    for row in message_rows:
        start = len(raw_messages)
        row_messages = get_message_row_model_message_payloads(row)
        raw_messages.extend(row_messages)
        row_message_ranges.append((start, len(raw_messages)))
    if not raw_messages:
        return [], row_message_ranges
    return list(ModelMessagesTypeAdapter.validate_python(raw_messages)), row_message_ranges


def expand_message_row_metadata(
    message_rows: Sequence[AIMessage],
    row_message_ranges: Sequence[tuple[int, int]],
) -> tuple[list[int | None], list[int | None], list[str | None], list[int | None]]:
    """
    展开消息行元信息到模型消息粒度

    :param message_rows: 消息行
    :param row_message_ranges: 行到模型消息范围映射
    :return:
    """
    message_ids: list[int | None] = []
    provider_ids: list[int | None] = []
    model_ids: list[str | None] = []
    message_indexes: list[int | None] = []
    for row, (start, end) in zip(message_rows, row_message_ranges, strict=False):
        count = end - start
        message_ids.extend([row.id] * count)
        provider_ids.extend([row.provider_id] * count)
        model_ids.extend([row.model_id] * count)
        message_indexes.extend([row.message_index] * count)
    return message_ids, provider_ids, model_ids, message_indexes


def get_row_model_messages(
    *,
    model_messages: Sequence[StoredModelMessage],
    row_message_ranges: Sequence[tuple[int, int]],
    row_index: int,
) -> list[StoredModelMessage]:
    """
    获取指定消息行对应的原始模型消息

    :param model_messages: 展开后的模型消息
    :param row_message_ranges: 行到模型消息范围映射
    :param row_index: 行索引
    :return:
    """
    start, end = row_message_ranges[row_index]
    return list(model_messages[start:end])
