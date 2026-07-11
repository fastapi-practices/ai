from dataclasses import replace
from typing import Any

from pydantic_ai import AgentRunResult, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import CompletionPersistenceContext, RegenerationPersistenceContext
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.protocol.base import ChatModelMessage
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title
from backend.plugin.ai.utils.message_storage import build_chat_message_record


def extract_assistant_run_messages(result: AgentRunResult[Any]) -> list[ChatModelMessage]:
    """
    提取本轮助手消息

    用户输入由聊天服务预先持久化，因此从首个模型响应开始提取，并移除运行期用户和系统提示

    :param result: Agent 运行结果
    :return:
    """
    new_messages = result.new_messages()
    first_response_index = next(
        (index for index, message in enumerate(new_messages) if isinstance(message, ModelResponse)),
        None,
    )
    if first_response_index is None:
        return []

    messages: list[ChatModelMessage] = []
    for message in new_messages[first_response_index:]:
        if isinstance(message, ModelRequest):
            parts = [part for part in message.parts if not isinstance(part, UserPromptPart | SystemPromptPart)]
            if not parts:
                continue
            message = replace(message, parts=parts)
        messages.append(message)
    return messages


def _build_chat_message_records(
    *,
    messages: list[ChatModelMessage],
    payload_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    按用户可见聊天消息构建持久化载荷

    :param messages: 原始模型消息
    :param payload_messages: 原始模型消息 JSON
    :return:
    """
    chat_message_records: list[dict[str, Any]] = []
    assistant_messages: list[dict[str, Any]] = []

    def flush_assistant_messages() -> None:
        if not assistant_messages:
            return
        chat_message_records.append(build_chat_message_record(role='assistant', model_messages=assistant_messages))
        assistant_messages.clear()

    for message, payload_message in zip(messages, payload_messages, strict=False):
        if isinstance(message, ModelRequest) and bool(message.parts) and isinstance(message.parts[0], UserPromptPart):
            flush_assistant_messages()
            chat_message_records.append(build_chat_message_record(role='user', model_messages=[payload_message]))
            continue
        assistant_messages.append(payload_message)

    flush_assistant_messages()
    return chat_message_records


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
    chat_message_records = _build_chat_message_records(
        messages=messages,
        payload_messages=payload_messages,
    )
    if persistence.assistant_message_id is not None:
        assistant_records = [record for record in chat_message_records if record['role'] == 'assistant']
        if not assistant_records:
            return
        assistant_record = assistant_records[-1]
        await ai_message_dao.update(
            db,
            persistence.assistant_message_id,
            {
                'provider_id': persistence.forwarded_props.provider_id,
                'model_id': persistence.forwarded_props.model_id,
                'status': AIMessageStatus.success,
                **assistant_record,
            },
        )
        return

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
                **record,
            }
            for offset, record in enumerate(chat_message_records)
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
    chat_message_records = _build_chat_message_records(
        messages=messages,
        payload_messages=payload_messages,
    )

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
        message_index_offset = len(chat_message_records) - old_message_count
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
            len(chat_message_records),
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
                **record,
            }
            for offset, record in enumerate(chat_message_records)
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
            if persistence.assistant_message_id is not None:
                payload_messages = to_jsonable_python([error_response], by_alias=True)
                assert isinstance(payload_messages, list)
                await ai_message_dao.update(
                    db,
                    persistence.assistant_message_id,
                    {
                        'provider_id': persistence.forwarded_props.provider_id,
                        'model_id': persistence.forwarded_props.model_id,
                        'role': 'assistant',
                        'status': AIMessageStatus.error,
                        'model_messages': payload_messages,
                    },
                )
            else:
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
