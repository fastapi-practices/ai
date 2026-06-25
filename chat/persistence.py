from pydantic_ai import ModelResponse, TextPart
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import CompletionPersistenceContext, RegenerationPersistenceContext
from backend.plugin.ai.protocol.base import ChatModelMessage
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title


async def persist_completion(
    *,
    db: AsyncSession,
    persistence: CompletionPersistenceContext,
    messages: list[ChatModelMessage],
) -> None:
    """
    持久化完成消息

    :param db: 数据库会话
    :param persistence: 持久化上下文
    :param messages: 待持久化消息
    :return:
    """
    if not messages:
        return
    payload_messages = to_jsonable_python(messages, by_alias=True)
    assert isinstance(payload_messages, list)

    current = await ai_conversation_dao.get_by_conversation_id_for_update(db, persistence.conversation_id)
    normalized_title = normalize_generated_conversation_title(title=persistence.title)
    if current:
        if current.user_id != persistence.user_id:
            raise errors.NotFoundError(msg='对话不存在')
        await ai_conversation_dao.update(
            db,
            current.id,
            UpdateAIConversationParam(
                conversation_id=current.conversation_id,
                title=normalized_title,
                provider_id=persistence.forwarded_props.provider_id,
                model_id=persistence.forwarded_props.model_id,
                user_id=current.user_id,
                pinned_time=current.pinned_time,
                context_start_message_id=current.context_start_message_id,
                context_cleared_time=current.context_cleared_time,
            ),
        )
    else:
        await ai_conversation_dao.create(
            db,
            CreateAIConversationParam(
                conversation_id=persistence.conversation_id,
                title=normalized_title,
                provider_id=persistence.forwarded_props.provider_id,
                model_id=persistence.forwarded_props.model_id,
                user_id=persistence.user_id,
            ),
        )

    next_message_index = await ai_message_dao.get_next_message_index(db, persistence.conversation_id)
    await ai_message_dao.bulk_create(
        db,
        [
            {
                'conversation_id': persistence.conversation_id,
                'provider_id': persistence.forwarded_props.provider_id,
                'model_id': persistence.forwarded_props.model_id,
                'message_index': next_message_index + offset,
                'message': message,
            }
            for offset, message in enumerate(payload_messages)
        ],
    )


async def persist_regeneration(
    *,
    db: AsyncSession,
    persistence: RegenerationPersistenceContext,
    messages: list[ChatModelMessage],
) -> None:
    """
    持久化重生成回复

    :param db: 数据库会话
    :param persistence: 重生成持久化上下文
    :param messages: 待持久化消息
    :return:
    """
    if not messages:
        return
    payload_messages = to_jsonable_python(messages, by_alias=True)
    assert isinstance(payload_messages, list)
    if not any(message.get('kind') == 'response' for message in payload_messages):
        return

    # 锁定当前用户对话，保护短事务写入顺序
    conversation = await ai_conversation_dao.get_by_conversation_id_for_update(db, persistence.conversation_id)
    if not conversation or conversation.user_id != persistence.user_id:
        raise errors.NotFoundError(msg='对话不存在')

    if persistence.replace_start_index is not None:
        replace_end_index = (
            persistence.replace_end_index
            if persistence.replace_end_index is not None
            else persistence.replace_start_index
        )
        await ai_message_dao.delete_message_index_range(
            db,
            persistence.conversation_id,
            persistence.replace_start_index,
            replace_end_index,
        )
        old_message_count = replace_end_index - persistence.replace_start_index + 1
        message_index_offset = len(payload_messages) - old_message_count
        if message_index_offset:
            await ai_message_dao.update_message_indexes_offset(
                db,
                persistence.conversation_id,
                replace_end_index + 1,
                message_index_offset,
            )
        message_index = persistence.replace_start_index
    elif persistence.insert_before_index is not None:
        await ai_message_dao.update_message_indexes_offset(
            db,
            persistence.conversation_id,
            persistence.insert_before_index,
            len(payload_messages),
        )
        message_index = persistence.insert_before_index
    else:
        message_index = await ai_message_dao.get_next_message_index(db, persistence.conversation_id)

    await ai_message_dao.bulk_create(
        db,
        [
            {
                'conversation_id': persistence.conversation_id,
                'provider_id': persistence.forwarded_props.provider_id,
                'model_id': persistence.forwarded_props.model_id,
                'message_index': message_index + offset,
                'message': message,
            }
            for offset, message in enumerate(payload_messages)
        ],
    )


async def persist_error_message(
    *,
    persistence: CompletionPersistenceContext,
    error_message: str,
) -> None:
    """
    回写模型请求失败消息

    :param persistence: 持久化上下文
    :param error_message: 错误信息
    :return:
    """
    raw_error_message = ' '.join(error_message.split()) if error_message else ''
    error_response = _build_error_response(
        model_id=persistence.forwarded_props.model_id,
        error_message=raw_error_message,
    )
    try:
        async with async_db_session.begin() as db:
            await persist_completion(db=db, persistence=persistence, messages=[error_response])
    except Exception as exc:
        log.exception(f'持久化聊天失败消息异常: {exc}')
    else:
        log.warning(f'聊天运行失败，已写入对话记录 conversation_id={persistence.conversation_id}: {raw_error_message}')


async def persist_regeneration_error_message(
    *,
    persistence: RegenerationPersistenceContext,
    error_message: str,
) -> None:
    """
    回写重生成失败消息

    :param persistence: 重生成持久化上下文
    :param error_message: 错误信息
    :return:
    """
    raw_error_message = ' '.join(error_message.split()) if error_message else ''
    error_response = _build_error_response(
        model_id=persistence.forwarded_props.model_id,
        error_message=raw_error_message,
    )
    try:
        async with async_db_session.begin() as db:
            await persist_regeneration(db=db, persistence=persistence, messages=[error_response])
    except Exception as exc:
        log.exception(f'持久化重生成失败消息异常: {exc}')
    else:
        log.warning(
            f'聊天重生成失败，已写入对话记录 conversation_id={persistence.conversation_id}: {raw_error_message}'
        )


def _build_error_response(*, model_id: str, error_message: str) -> ModelResponse:
    """
    构建模型错误回复

    :param model_id: 模型 ID
    :param error_message: 错误信息
    :return:
    """
    display_error = error_message or '模型请求失败，请稍后重试'
    return ModelResponse(
        parts=[TextPart(content=f'模型请求失败：{display_error}')],
        model_name=model_id,
        metadata={'is_error': True, 'error_message': display_error},
    )
