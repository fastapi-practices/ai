from collections.abc import Sequence

from pydantic_ai import (
    AudioUrl,
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
    VideoUrl,
)

from backend.plugin.ai.enums import (
    AIChatAttachmentSourceType,
    AIChatAttachmentType,
    AIMessageBlockType,
    AIMessageRoleType,
    AIMessageType,
)
from backend.plugin.ai.schema.message import (
    GetAIMessageBlockDetail,
    GetAIMessageDetail,
)
from backend.utils.timezone import timezone


def get_attachment_type(content: BinaryContent) -> AIChatAttachmentType:
    """
    获取二进制附件类型

    :param content: 二进制内容
    :return:
    """
    if content.is_image:
        return AIChatAttachmentType.image
    if content.is_audio:
        return AIChatAttachmentType.audio
    if content.is_video:
        return AIChatAttachmentType.video
    return AIChatAttachmentType.document


def build_file_block(
    attachment: ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent,
) -> GetAIMessageBlockDetail:
    """
    构建文件内容块

    :param attachment: 附件
    :return:
    """
    attachment_identifier = attachment.identifier
    attachment_name = ((attachment.vendor_metadata or {}).get('filename')) or attachment_identifier
    if isinstance(attachment, BinaryContent):
        file_type = get_attachment_type(attachment)
        source_type = AIChatAttachmentSourceType.base64
        url = attachment.data_uri
    else:
        file_type = {
            ImageUrl: AIChatAttachmentType.image,
            AudioUrl: AIChatAttachmentType.audio,
            VideoUrl: AIChatAttachmentType.video,
            DocumentUrl: AIChatAttachmentType.document,
        }[type(attachment)]
        source_type = AIChatAttachmentSourceType.url
        url = attachment.url

    return GetAIMessageBlockDetail(
        type=AIMessageBlockType.file,
        file_type=file_type,
        source_type=source_type,
        mime_type=attachment.media_type,
        name=attachment_name,
        url=url,
    )


def build_text_block(*, type_: AIMessageBlockType, text: str) -> GetAIMessageBlockDetail | None:
    """
    构建文本内容块

    :param type_: 内容块类型
    :param text: 文本
    :return:
    """
    normalized_text = text.strip()
    if not normalized_text:
        return None
    return GetAIMessageBlockDetail(type=type_, text=normalized_text)


def serialize_messages(  # noqa: C901
    messages: Sequence[ModelMessage],
    *,
    conversation_id: str | None = None,
    message_ids: Sequence[int | None] | None = None,
    provider_ids: Sequence[int | None] | None = None,
    model_ids: Sequence[str | None] | None = None,
) -> list[GetAIMessageDetail]:
    """
    序列化模型消息

    :param messages: 模型消息序列
    :param conversation_id: 对话 ID
    :param message_ids: 消息 ID 序列
    :return:
    """
    parsed_messages: list[GetAIMessageDetail] = []
    for model_message_index, message in enumerate(messages):
        message_id = message_ids[model_message_index] if message_ids else None
        provider_id = provider_ids[model_message_index] if provider_ids else None
        model_id = model_ids[model_message_index] if model_ids else None

        if isinstance(message, ModelRequest) and message.parts:
            first_part = message.parts[0]
            if isinstance(first_part, UserPromptPart):
                blocks: list[GetAIMessageBlockDetail] = []
                if isinstance(first_part.content, str):
                    text_block = build_text_block(type_=AIMessageBlockType.text, text=first_part.content)
                    if text_block:
                        blocks.append(text_block)
                else:
                    for item in first_part.content:
                        if isinstance(item, str):
                            text_block = build_text_block(type_=AIMessageBlockType.text, text=item)
                            if text_block:
                                blocks.append(text_block)
                            continue
                        if isinstance(item, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent)):
                            blocks.append(build_file_block(item))
                parsed_messages.append(
                    GetAIMessageDetail(
                        message_id=message_id,
                        message_index=len(parsed_messages),
                        role=AIMessageRoleType.user,
                        message_type=AIMessageType.normal,
                        created_time=first_part.timestamp.isoformat(),
                        provider_id=provider_id,
                        model_id=model_id,
                        blocks=blocks,
                        conversation_id=conversation_id,
                    )
                )
            continue

        if not isinstance(message, ModelResponse):
            continue

        created_time = message.timestamp.isoformat() if message.timestamp else timezone.now().isoformat()
        message_type = AIMessageType.error if (message.metadata or {}).get('is_error') else AIMessageType.normal
        blocks: list[GetAIMessageBlockDetail] = []
        for part in message.parts:
            if isinstance(part, ThinkingPart):
                reasoning_block = build_text_block(type_=AIMessageBlockType.reasoning, text=part.content)
                if reasoning_block:
                    blocks.append(reasoning_block)
                continue
            if isinstance(part, TextPart):
                text_block = build_text_block(type_=AIMessageBlockType.text, text=part.content)
                if text_block:
                    blocks.append(text_block)
                continue
            if isinstance(part, FilePart):
                blocks.append(build_file_block(part.content))

        parsed_messages.append(
            GetAIMessageDetail(
                message_id=message_id,
                message_index=len(parsed_messages),
                role=AIMessageRoleType.assistant,
                message_type=message_type,
                created_time=created_time,
                provider_id=provider_id,
                model_id=model_id or message.model_name,
                blocks=blocks,
                conversation_id=conversation_id,
            )
        )

    return parsed_messages
