from collections.abc import Sequence

from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from backend.plugin.ai.enums import AIMessageRoleType
from backend.plugin.ai.schema.message import GetAIMessageAttachmentDetail, GetAIMessageDetail
from backend.utils.timezone import timezone


def to_messages(  # noqa: C901
    messages: Sequence[ModelMessage],
    *,
    conversation_id: str | None = None,
    message_ids: Sequence[int | None] | None = None,
) -> list[GetAIMessageDetail]:
    """
    将模型消息序列转换为前端消息列表

    :param messages: 模型消息序列
    :param conversation_id: 对话 ID
    :param message_ids: 消息 ID 序列
    :return:
    """
    parsed_messages: list[GetAIMessageDetail] = []
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
            if isinstance(first_part, UserPromptPart):
                attachments: list[GetAIMessageAttachmentDetail] = []
                text_parts: list[str] = []
                if isinstance(first_part.content, str):
                    text_parts.append(first_part.content)
                else:
                    for item in first_part.content:
                        if isinstance(item, str):
                            text_parts.append(item)
                            continue
                        if isinstance(item, ImageUrl):
                            attachments.append(
                                GetAIMessageAttachmentDetail(
                                    type='image',
                                    mime_type=item.media_type,
                                    url=item.url,
                                )
                            )
                            continue
                        if isinstance(item, DocumentUrl):
                            attachments.append(
                                GetAIMessageAttachmentDetail(
                                    type='document',
                                    mime_type=item.media_type,
                                    url=item.url,
                                )
                            )
                            continue
                        if isinstance(item, BinaryContent):
                            attachments.append(
                                GetAIMessageAttachmentDetail(
                                    type='image' if item.is_image else 'document',
                                    mime_type=item.media_type,
                                    url=item.data_uri,
                                )
                            )
                parsed_messages.append(
                    GetAIMessageDetail(
                        message_id=message_id,
                        message_index=len(parsed_messages),
                        role=AIMessageRoleType.user,
                        timestamp=first_part.timestamp.isoformat(),
                        content=' '.join(text_parts).strip(),
                        attachments=attachments,
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
            role: AIMessageRoleType | None = None
            content = ''

            if isinstance(part, ThinkingPart):
                role = AIMessageRoleType.thinking
                content = part.content
            elif isinstance(part, TextPart):
                role = AIMessageRoleType.model
                content = part.content
            if role is None:
                if isinstance(part, FilePart):
                    parsed_messages.append(
                        GetAIMessageDetail(
                            message_id=message_id,
                            message_index=len(parsed_messages),
                            role=AIMessageRoleType.model,
                            timestamp=timestamp,
                            content='',
                            attachments=[
                                GetAIMessageAttachmentDetail(
                                    type='image' if part.content.is_image else 'document',
                                    mime_type=part.content.media_type,
                                    url=part.content.data_uri,
                                )
                            ],
                            conversation_id=conversation_id,
                            is_error=is_error,
                            error_message=error_message,
                            structured_data=structured_data,
                        )
                    )
                continue
            parsed_messages.append(
                GetAIMessageDetail(
                    message_id=message_id,
                    message_index=len(parsed_messages),
                    role=role,
                    timestamp=timestamp,
                    content=content,
                    attachments=[],
                    conversation_id=conversation_id,
                    is_error=is_error,
                    error_message=error_message,
                    structured_data=structured_data,
                )
            )
    return parsed_messages
