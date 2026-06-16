from pydantic_ai import ModelResponse, TextPart
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.dataclasses import CompletionPersistence
from backend.plugin.ai.protocol.base import ChatModelMessage
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title


async def persist_completion(
    *,
    db: AsyncSession,
    persistence: CompletionPersistence,
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

    await _upsert_conversation(db=db, persistence=persistence)
    await persistence.strategy.apply(
        db=db,
        conversation_id=persistence.conversation_id,
        payload_messages=payload_messages,
        forwarded_props=persistence.forwarded_props,
    )


async def persist_error_message(
    *,
    persistence: CompletionPersistence,
    error_message: str,
) -> None:
    """
    回写模型请求失败消息

    :param persistence: 持久化上下文
    :param error_message: 错误信息
    :return:
    """
    raw_error_message = ' '.join(error_message.split()) if error_message else ''
    display_error = raw_error_message or '模型请求失败，请稍后重试'
    error_response = ModelResponse(
        parts=[TextPart(content=f'模型请求失败：{display_error}')],
        model_name=persistence.forwarded_props.model_id,
        metadata={'is_error': True, 'error_message': display_error},
    )
    try:
        async with async_db_session.begin() as db:
            await persist_completion(db=db, persistence=persistence, messages=[error_response])
    except Exception as exc:
        log.exception(f'持久化聊天失败消息异常: {exc}')
    else:
        log.warning(f'聊天运行失败，已写入对话记录 conversation_id={persistence.conversation_id}: {raw_error_message}')


async def _upsert_conversation(
    *,
    db: AsyncSession,
    persistence: CompletionPersistence,
) -> None:
    """
    更新或创建对话记录

    :param db: 数据库会话
    :param persistence: 持久化上下文
    :return:
    """
    current = persistence.conversation or await ai_conversation_dao.get_by_conversation_id(
        db, persistence.conversation_id
    )
    normalized_title = normalize_generated_conversation_title(title=persistence.title)
    if current:
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
        return
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
