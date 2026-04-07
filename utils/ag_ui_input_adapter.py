from base64 import b64decode
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from ag_ui.core import (
    AudioInputContent,
    BinaryInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentDataSource,
    TextInputContent,
    UserMessage,
    VideoInputContent,
)
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)

from backend.common.exception import errors
from backend.database.db import uuid4_str

if TYPE_CHECKING:
    from pydantic_ai.messages import UploadedFileProviderName

PromptContentItem: TypeAlias = str | AudioUrl | BinaryContent | DocumentUrl | ImageUrl | UploadedFile | VideoUrl
UserPromptContent: TypeAlias = str | Sequence[PromptContentItem]
MediaInputPart: TypeAlias = ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent


def build_binary_content(
    *,
    data: bytes,
    media_type: str,
    identifier: str | None,
    vendor_metadata: dict[str, Any] | None,
) -> BinaryContent:
    """
    构建二进制内容

    :param data: 二进制数据
    :param media_type: 媒体类型
    :param identifier: 标识
    :param vendor_metadata: 扩展元数据
    :return:
    """
    return BinaryContent.narrow_type(
        BinaryContent(
            data=data,
            media_type=media_type,
            identifier=identifier,
            vendor_metadata=vendor_metadata,
        )
    )


def build_file_url_content(
    *,
    url: str,
    media_type: str,
    identifier: str | None,
    vendor_metadata: dict[str, Any] | None,
) -> AudioUrl | DocumentUrl | ImageUrl | VideoUrl:
    """
    构建文件 URL 内容

    :param url: 文件地址
    :param media_type: 媒体类型
    :param identifier: 标识
    :param vendor_metadata: 扩展元数据
    :return:
    """
    constructor = {'image': ImageUrl, 'video': VideoUrl, 'audio': AudioUrl}.get(
        media_type.split('/', 1)[0], DocumentUrl
    )
    return constructor(
        url=url,
        media_type=media_type,
        identifier=identifier,
        vendor_metadata=vendor_metadata,
    )


def deserialize_binary_input_part(part: BinaryInputContent) -> PromptContentItem:
    """
    解析二进制输入片段

    :param part: 二进制输入片段
    :return:
    """
    extra = part.model_extra if isinstance(part.model_extra, dict) else {}
    raw_vendor_metadata = extra.get('vendorMetadata')
    vendor_metadata = dict(raw_vendor_metadata) if isinstance(raw_vendor_metadata, dict) else {}
    if part.filename:
        vendor_metadata['filename'] = part.filename
    provider_name = extra.get('providerName')
    if part.id and isinstance(provider_name, str):
        return UploadedFile(
            file_id=part.id,
            provider_name=cast('UploadedFileProviderName', provider_name),
            media_type=part.mime_type,
            identifier=extra.get('identifier') if isinstance(extra.get('identifier'), str) else None,
            vendor_metadata=vendor_metadata or None,
        )
    if part.url:
        try:
            parsed_binary = BinaryContent.from_data_uri(part.url)
        except ValueError:
            return build_file_url_content(
                url=part.url,
                media_type=part.mime_type,
                identifier=part.id,
                vendor_metadata=vendor_metadata or None,
            )
        return build_binary_content(
            data=parsed_binary.data,
            media_type=parsed_binary.media_type,
            identifier=part.id,
            vendor_metadata=vendor_metadata or None,
        )
    if part.data:
        return build_binary_content(
            data=b64decode(part.data),
            media_type=part.mime_type,
            identifier=part.id,
            vendor_metadata=vendor_metadata or None,
        )
    raise errors.RequestError(msg='聊天消息格式非法')


def deserialize_media_input_part(part: MediaInputPart) -> PromptContentItem:
    """
    解析媒体输入片段

    :param part: 媒体输入片段
    :return:
    """
    metadata = part.metadata if isinstance(part.metadata, dict) else {}
    mime_type = (
        part.source.mime_type
        or {
            ImageInputContent: 'image/*',
            AudioInputContent: 'audio/*',
            VideoInputContent: 'video/*',
            DocumentInputContent: 'application/octet-stream',
        }[type(part)]
    )
    attachment_id = metadata.get('id') if isinstance(metadata.get('id'), str) else uuid4_str()
    raw_vendor_metadata = metadata.get('vendorMetadata')
    vendor_metadata = dict(raw_vendor_metadata) if isinstance(raw_vendor_metadata, dict) else {}
    if isinstance(metadata.get('filename'), str):
        vendor_metadata['filename'] = metadata['filename']
    if isinstance(part.source, InputContentDataSource):
        return build_binary_content(
            data=b64decode(part.source.value),
            media_type=mime_type,
            identifier=attachment_id,
            vendor_metadata=vendor_metadata or None,
        )
    try:
        parsed_binary = BinaryContent.from_data_uri(part.source.value)
    except ValueError:
        media_type = part.source.mime_type or mime_type
        return build_file_url_content(
            url=part.source.value,
            media_type=media_type,
            identifier=attachment_id,
            vendor_metadata=vendor_metadata or None,
        )
    return build_binary_content(
        data=parsed_binary.data,
        media_type=parsed_binary.media_type,
        identifier=attachment_id,
        vendor_metadata=vendor_metadata or None,
    )


def deserialize_current_user_message(message: UserMessage) -> ModelRequest:
    """
    解析当前轮用户消息，保留文件标识和文件名

    :param message: 用户消息
    :return:
    """
    content = message.content
    if isinstance(content, str):
        return ModelRequest(parts=[UserPromptPart(content=content)])

    user_prompt_content: list[PromptContentItem] = []
    for part in content:
        if isinstance(part, TextInputContent):
            user_prompt_content.append(part.text)
            continue
        if isinstance(part, BinaryInputContent):
            user_prompt_content.append(deserialize_binary_input_part(part))
            continue
        if isinstance(part, (ImageInputContent, AudioInputContent, VideoInputContent, DocumentInputContent)):
            user_prompt_content.append(deserialize_media_input_part(part))
            continue
        raise errors.RequestError(msg='聊天消息格式非法')

    if not user_prompt_content:
        raise errors.RequestError(msg='聊天消息不能为空')

    user_prompt: UserPromptContent
    if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str):
        user_prompt = cast('str', user_prompt_content[0])
    else:
        user_prompt = user_prompt_content
    return ModelRequest(parts=[UserPromptPart(content=user_prompt)])
