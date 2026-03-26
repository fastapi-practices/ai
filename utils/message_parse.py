from collections.abc import Sequence

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from backend.plugin.ai.enums import AIChatMessageRoleType
from backend.plugin.ai.schema.chat import GetAIChatMessageDetail
from backend.utils.timezone import timezone


def to_chat_messages(
    messages: Sequence[ModelMessage],
    *,
    conversation_id: str | None = None,
    message_ids: Sequence[int | None] | None = None,
) -> list[GetAIChatMessageDetail]:
    """
    将模型消息序列转换为前端消息列表

    :param messages: 模型消息序列
    :param conversation_id: 会话 ID
    :param message_ids: 消息 ID 序列
    :return:
    """
    chat_messages: list[GetAIChatMessageDetail] = []
    for model_message_index, message in enumerate(messages):
        message_id = message_ids[model_message_index] if message_ids else None
        metadata = message.metadata or {}
        is_error = bool(metadata.get('is_error', False))
        error_message = metadata.get('error_message')
        structured_data = metadata.get('structured_data')
        if error_message is not None:
            error_message = str(error_message)

        if isinstance(message, ModelRequest) and message.parts:
            first_part = message.parts[0]
            if isinstance(first_part, UserPromptPart) and isinstance(first_part.content, str):
                chat_messages.append(
                    GetAIChatMessageDetail(
                        message_id=message_id,
                        message_index=len(chat_messages),
                        role=AIChatMessageRoleType.user,
                        timestamp=first_part.timestamp.isoformat(),
                        content=first_part.content,
                        conversation_id=conversation_id,
                        is_error=is_error,
                        error_message=error_message,
                        structured_data=structured_data,
                    )
                )
            continue

        if not isinstance(message, ModelResponse):
            continue

        timestamp = message.timestamp.isoformat() if message.timestamp else timezone.now().isoformat()
        for part in message.parts:
            role: AIChatMessageRoleType | None = None
            content = ''

            if isinstance(part, ThinkingPart):
                role = AIChatMessageRoleType.thinking
                content = part.content
            elif isinstance(part, TextPart):
                role = AIChatMessageRoleType.model
                content = part.content
            if role is None:
                continue
            chat_messages.append(
                GetAIChatMessageDetail(
                    message_id=message_id,
                    message_index=len(chat_messages),
                    role=role,
                    timestamp=timestamp,
                    content=content,
                    conversation_id=conversation_id,
                    is_error=is_error,
                    error_message=error_message,
                    structured_data=structured_data,
                )
            )
    return chat_messages
