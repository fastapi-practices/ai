from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_core import to_jsonable_python

from backend.common.exception import errors
from backend.utils.timezone import timezone


class ChatMessage(TypedDict):
    """发送给浏览器的消息格式"""

    message_index: int
    role: Literal['user', 'model']
    timestamp: str
    content: str
    conversation_id: str | None
    is_error: bool
    error_message: str | None


@dataclass(slots=True)
class ChatTranscriptItem:
    """聊天转录项"""

    message_index: int
    model_message_index: int
    role: Literal['user', 'model']
    timestamp: str
    content: str
    conversation_id: str | None
    is_error: bool = False
    error_message: str | None = None

    def to_chat_message(self) -> ChatMessage:
        return {
            'message_index': self.message_index,
            'role': self.role,
            'timestamp': self.timestamp,
            'content': self.content,
            'conversation_id': self.conversation_id,
            'is_error': self.is_error,
            'error_message': self.error_message,
        }


def make_chat_message(
    *,
    message_index: int,
    role: Literal['user', 'model'],
    content: str,
    timestamp: datetime | None = None,
    conversation_id: str | None = None,
    is_error: bool = False,
    error_message: str | None = None,
) -> ChatMessage:
    return {
        'message_index': message_index,
        'role': role,
        'timestamp': (timestamp or timezone.now()).isoformat(),
        'content': content,
        'conversation_id': conversation_id,
        'is_error': is_error,
        'error_message': error_message,
    }


def to_chat_message(
    message: ModelMessage,
    *,
    message_index: int,
    conversation_id: str | None = None,
) -> ChatMessage:
    metadata = message.metadata or {}
    is_error = bool(metadata.get('is_error', False))
    error_message = metadata.get('error_message')
    if error_message is not None:
        error_message = str(error_message)

    first_part = message.parts[0]
    if isinstance(message, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return make_chat_message(
                message_index=message_index,
                role='user',
                timestamp=first_part.timestamp,
                content=first_part.content,
                conversation_id=conversation_id,
                is_error=is_error,
                error_message=error_message,
            )
    elif isinstance(message, ModelResponse) and isinstance(first_part, TextPart):
        return make_chat_message(
            message_index=message_index,
            role='model',
            timestamp=message.timestamp,
            content=first_part.content,
            conversation_id=conversation_id,
            is_error=is_error,
            error_message=error_message,
        )
    raise errors.NotFoundError(msg=f'消息类型错误: {message}')


def build_chat_transcript(
    messages: Sequence[ModelMessage],
    *,
    conversation_id: str | None = None,
) -> list[ChatTranscriptItem]:
    transcript: list[ChatTranscriptItem] = []
    for model_message_index, message in enumerate(messages):
        try:
            parsed_message = to_chat_message(
                message,
                message_index=len(transcript),
                conversation_id=conversation_id,
            )
        except errors.NotFoundError:
            parsed_message = None
        if parsed_message is not None:
            transcript.append(
                ChatTranscriptItem(
                    message_index=parsed_message['message_index'],
                    model_message_index=model_message_index,
                    role=parsed_message['role'],
                    timestamp=parsed_message['timestamp'],
                    content=parsed_message['content'],
                    conversation_id=parsed_message['conversation_id'],
                    is_error=parsed_message['is_error'],
                    error_message=parsed_message['error_message'],
                )
            )
    return transcript


def to_chat_messages(messages: Sequence[ModelMessage], *, conversation_id: str | None = None) -> list[ChatMessage]:
    return [item.to_chat_message() for item in build_chat_transcript(messages, conversation_id=conversation_id)]


def get_chat_transcript_item(
    messages: Sequence[ModelMessage],
    *,
    message_index: int,
    conversation_id: str | None = None,
) -> ChatTranscriptItem:
    transcript = build_chat_transcript(messages, conversation_id=conversation_id)
    if message_index < 0 or message_index >= len(transcript):
        raise errors.NotFoundError(msg='聊天消息不存在')
    return transcript[message_index]


def truncate_model_messages_by_index(
    messages: Sequence[ModelMessage],
    *,
    message_index: int,
    conversation_id: str | None = None,
) -> list[ModelMessage]:
    target_item = get_chat_transcript_item(messages, message_index=message_index, conversation_id=conversation_id)
    return list(messages[: target_item.model_message_index])


def delete_model_message_by_index(
    messages: Sequence[ModelMessage],
    *,
    message_index: int,
    conversation_id: str | None = None,
) -> list[ModelMessage]:
    target_item = get_chat_transcript_item(messages, message_index=message_index, conversation_id=conversation_id)
    remaining_messages = list(messages)
    del remaining_messages[target_item.model_message_index]
    return remaining_messages


def parse_model_messages(messages: object | None) -> list[ModelMessage]:
    if not messages:
        return []
    return ModelMessagesTypeAdapter.validate_python(messages)


def serialize_model_messages(messages: Sequence[ModelMessage]) -> list[dict[str, object]]:
    payload = to_jsonable_python(list(messages))
    assert isinstance(payload, list)
    return payload
