from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pydantic_ai import AgentRunResult, ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart
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


def extract_assistant_messages(run_messages: Sequence[ChatModelMessage]) -> list[ChatModelMessage]:
    """
    提取当前轮助手消息

    用户输入由聊天服务预先持久化，因此从首个模型响应开始提取，并移除运行期用户和系统提示

    :param run_messages: 当前轮原始模型消息
    :return:
    """
    first_response_index = next(
        (index for index, message in enumerate(run_messages) if isinstance(message, ModelResponse)),
        None,
    )
    if first_response_index is None:
        return []

    assistant_messages: list[ChatModelMessage] = []
    for message in run_messages[first_response_index:]:
        if isinstance(message, ModelRequest):
            parts = [part for part in message.parts if not isinstance(part, UserPromptPart | SystemPromptPart)]
            if not parts:
                continue
            message = replace(message, parts=parts)
        assistant_messages.append(message)
    return assistant_messages


def extract_assistant_run_messages(result: AgentRunResult[Any]) -> list[ChatModelMessage]:
    """
    提取成功运行中的当前轮助手消息

    :param result: Agent 运行结果
    :return:
    """
    return extract_assistant_messages(result.new_messages())


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
    status: AIMessageStatus = AIMessageStatus.success,
) -> None:
    """
    持久化完成消息

    :param db: 数据库会话
    :param persistence: 持久化上下文
    :param messages: 待持久化消息
    :param status: 消息状态
    :return:
    """
    if not messages:
        if persistence.assistant_message_id is not None:
            await _finalize_pending_placeholder(
                db=db,
                message_id=persistence.assistant_message_id,
                payload={'status': status},
            )
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
            await _finalize_pending_placeholder(
                db=db,
                message_id=persistence.assistant_message_id,
                payload={'status': status},
            )
            return
        assistant_record = assistant_records[-1]
        await _finalize_pending_placeholder(
            db=db,
            message_id=persistence.assistant_message_id,
            payload={
                'provider_id': persistence.forwarded_props.provider_id,
                'model_id': persistence.forwarded_props.model_id,
                'status': status,
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
                'status': status,
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
    status: AIMessageStatus = AIMessageStatus.success,
) -> None:
    """
    持久化重生成回复

    :param db: 数据库会话
    :param persistence: 重生成持久化上下文
    :param messages: 待持久化消息
    :param status: 消息状态
    :return:
    """
    conversation = await ai_conversation_dao.get_by_conversation_id_for_update(db, persistence.conversation_id)
    if not conversation or conversation.user_id != persistence.user_id:
        raise errors.NotFoundError(msg='对话不存在')
    assistant_message_id = persistence.assistant_message_id
    if assistant_message_id is None:
        raise RuntimeError('缺少重生成占位消息')
    assistant_placeholder = await ai_message_dao.get(db, assistant_message_id)
    if (
        assistant_placeholder is None
        or assistant_placeholder.conversation_id != persistence.conversation_id
        or assistant_placeholder.status != AIMessageStatus.pending
    ):
        raise errors.ConflictError(msg='重生成任务已失效，请重试')
    if not any(isinstance(message, ModelResponse) for message in messages):
        if status == AIMessageStatus.success:
            await _delete_pending_placeholder(db=db, message_id=assistant_message_id)
        else:
            await _finalize_pending_placeholder(
                db=db,
                message_id=assistant_message_id,
                payload={'status': status},
            )
        return
    payload_messages = to_jsonable_python(messages, by_alias=True)
    assert isinstance(payload_messages, list)
    chat_message_records = _build_chat_message_records(
        messages=messages,
        payload_messages=payload_messages,
    )

    await _delete_pending_placeholder(db=db, message_id=assistant_message_id)

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
                'status': status,
                **record,
            }
            for offset, record in enumerate(chat_message_records)
        ],
    )


async def persist_terminal_completion(
    *,
    persistence: CompletionPersistenceContext,
    messages: list[ChatModelMessage],
    status: AIMessageStatus,
    reason: str = '',
) -> None:
    """
    回写普通聊天非成功终态

    :param persistence: 持久化上下文
    :param messages: Pydantic AI 捕获的助手消息
    :param status: 消息终态
    :param reason: 终止原因
    :return:
    """
    raw_reason = ' '.join(reason.split()) if reason else ''
    try:
        async with async_db_session.begin() as db:
            await persist_completion(
                db=db,
                persistence=persistence,
                messages=messages,
                status=status,
            )
    except Exception as exc:
        log.exception(f'持久化聊天终态消息异常: {exc}')
        await _mark_placeholder_terminal(
            message_id=persistence.assistant_message_id,
            status=status,
        )
    else:
        status_label = '失败' if status == AIMessageStatus.error else '中断'
        reason_suffix = f': {raw_reason}' if raw_reason else ''
        log.warning(
            f'聊天运行{status_label}，已写入对话记录 conversation_id={persistence.conversation_id}{reason_suffix}'
        )


async def persist_terminal_regeneration(
    *,
    persistence: RegenerationPersistenceContext,
    messages: list[ChatModelMessage],
    status: AIMessageStatus,
    reason: str = '',
) -> None:
    """
    回写重生成非成功终态

    :param persistence: 重生成持久化上下文
    :param messages: Pydantic AI 捕获的助手消息
    :param status: 消息终态
    :param reason: 终止原因
    :return:
    """
    raw_reason = ' '.join(reason.split()) if reason else ''
    try:
        async with async_db_session.begin() as db:
            await persist_regeneration(
                db=db,
                persistence=persistence,
                messages=messages,
                status=status,
            )
    except Exception as exc:
        log.exception(f'持久化重生成终态消息异常: {exc}')
        await _mark_placeholder_terminal(
            message_id=persistence.assistant_message_id,
            status=status,
        )
    else:
        status_label = '失败' if status == AIMessageStatus.error else '中断'
        reason_suffix = f': {raw_reason}' if raw_reason else ''
        log.warning(
            f'聊天重生成{status_label}，已写入对话记录 conversation_id={persistence.conversation_id}{reason_suffix}'
        )


async def _finalize_pending_placeholder(
    *,
    db: AsyncSession,
    message_id: int,
    payload: dict[str, Any],
) -> None:
    """通过待生成状态校验防止过期任务覆盖新终态"""
    count = await ai_message_dao.finalize_pending(db, message_id, payload)
    if count == 0:
        raise errors.ConflictError(msg='生成任务已失效，请重试')


async def _delete_pending_placeholder(*, db: AsyncSession, message_id: int) -> None:
    """通过待生成状态校验删除重生成占位消息"""
    count = await ai_message_dao.delete_pending(db, message_id)
    if count == 0:
        raise errors.ConflictError(msg='生成任务已失效，请重试')


async def _mark_placeholder_terminal(*, message_id: int | None, status: AIMessageStatus) -> None:
    """
    在主持久化失败后兜底释放生成占位消息

    :param message_id: 占位消息 ID
    :param status: 目标终态
    :return:
    """
    if message_id is None:
        return
    async with async_db_session.begin() as db:
        await ai_message_dao.finalize_pending(db, message_id, {'status': status})
