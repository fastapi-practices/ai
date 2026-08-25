from copy import deepcopy
from dataclasses import replace
from typing import Any

import anyio

from pydantic_ai import AgentRunResult, ModelRequest, ModelResponse, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.chat.persistence import (
    extract_assistant_messages,
    extract_assistant_run_messages,
    persist_regeneration,
    persist_terminal_regeneration,
)
from backend.plugin.ai.chat.runner import is_user_prompt_message, open_chat_session
from backend.plugin.ai.chat.runs import abort_prepared_run, activate_run
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import ChatConversationState, RegenerationPersistenceContext
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.model import AIMessage
from backend.plugin.ai.protocol.registry import get_chat_protocol_adapter
from backend.plugin.ai.schema.chat import AIChatRegenerateParam
from backend.plugin.ai.schema.conversation import UpdateAIConversationParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.message_storage import (
    expand_message_rows,
    get_message_row_model_message_payloads,
    get_row_model_messages,
)


class AIMessageService:
    """AI 消息服务"""

    @staticmethod
    def _is_user_message_row(
        *,
        model_messages: list[ModelRequest | ModelResponse],
        row_index: int,
        row_model_message_ranges: list[tuple[int, int]],
    ) -> bool:
        """
        判断是否为用户消息行

        :param model_messages: 模型消息列表
        :param row_index: 消息行索引
        :param row_model_message_ranges: 行到模型消息范围映射
        :return:
        """
        row_messages = get_row_model_messages(
            model_messages=model_messages,
            row_message_ranges=row_model_message_ranges,
            row_index=row_index,
        )
        return len(row_messages) == 1 and is_user_prompt_message(message=row_messages[0])

    @staticmethod
    def _get_message_row_index(*, message_rows: list[AIMessage], pk: int) -> int:
        """
        获取消息行索引

        :param message_rows: 消息行列表
        :param pk: 消息主键
        :return:
        """
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == pk), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='消息不存在，请刷新后重试')
        return message_row_index

    def _get_regenerate_target(self, *, state: ChatConversationState, pk: int) -> tuple[int, int]:
        """
        获取可重发的用户消息行及其模型消息结束下标

        :param state: 对话状态
        :param pk: 消息主键
        :return:
        """
        target_index = self._get_message_row_index(message_rows=state.message_rows, pk=pk)
        target_messages = get_row_model_messages(
            model_messages=state.model_messages,
            row_message_ranges=state.row_model_message_ranges,
            row_index=target_index,
        )
        if len(target_messages) != 1 or not is_user_prompt_message(message=target_messages[0]):
            raise errors.RequestError(msg='仅支持根据用户消息重发')
        last_user_index: int | None = None
        for index in range(len(state.message_rows)):
            if self._is_user_message_row(
                model_messages=state.model_messages,
                row_index=index,
                row_model_message_ranges=state.row_model_message_ranges,
            ):
                last_user_index = index
        if last_user_index is None:
            raise errors.RequestError(msg='当前对话没有可重发的用户消息')
        if target_index != last_user_index:
            raise errors.RequestError(msg='仅支持重发最后一条用户消息，请刷新后重试')
        _, target_end_index = state.row_model_message_ranges[target_index]
        return target_index, target_end_index

    def _get_reply_segment_indexes(
        self,
        *,
        message_rows: list[AIMessage],
        model_messages: list[ModelRequest | ModelResponse],
        reply_start_index: int,
        row_model_message_ranges: list[tuple[int, int]],
    ) -> tuple[int | None, int | None, int | None]:
        """
        获取回复段的消息索引范围

        :param message_rows: 消息行列表
        :param model_messages: 模型消息列表
        :param reply_start_index: 回复段起始行索引
        :param row_model_message_ranges: 行到模型消息范围映射
        :return:
        """
        if reply_start_index >= len(message_rows):
            return None, None, None
        if self._is_user_message_row(
            model_messages=model_messages,
            row_index=reply_start_index,
            row_model_message_ranges=row_model_message_ranges,
        ):
            return None, None, message_rows[reply_start_index].message_index
        reply_end_index = reply_start_index
        for index in range(reply_start_index + 1, len(message_rows)):
            if self._is_user_message_row(
                model_messages=model_messages,
                row_index=index,
                row_model_message_ranges=row_model_message_ranges,
            ):
                break
            reply_end_index = index
        return (
            message_rows[reply_start_index].message_index,
            message_rows[reply_end_index].message_index,
            None,
        )

    @staticmethod
    async def _update_last_user_message_content(
        *,
        db: AsyncSession,
        state: ChatConversationState,
        row_index: int,
        target_end_index: int,
        content: str | None,
    ) -> None:
        """
        重发时按需覆盖最后一条用户消息内容

        :param db: 数据库会话
        :param state: 对话状态
        :param row_index: 目标消息行索引
        :param target_end_index: 目标消息结束下标
        :param content: 新的用户消息内容，空值表示不覆盖
        :return:
        """
        if content is None:
            return
        target_messages = get_row_model_messages(
            model_messages=state.model_messages,
            row_message_ranges=state.row_model_message_ranges,
            row_index=row_index,
        )
        target_message = target_messages[0] if target_messages else None
        if len(target_messages) != 1 or not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not isinstance(target_message.parts[0].content, str):
            raise errors.RequestError(msg='当前消息暂不支持直接编辑')
        normalized_content = ' '.join(content.split())
        if not normalized_content:
            raise errors.RequestError(msg='消息内容不能为空')

        model_messages_payload = deepcopy(get_message_row_model_message_payloads(state.message_rows[row_index]))
        model_payload = deepcopy(model_messages_payload[0])
        model_payload['parts'][0]['content'] = normalized_content
        model_messages_payload[0] = model_payload
        await ai_message_dao.update(db, state.message_rows[row_index].id, {'model_messages': model_messages_payload})
        state.model_messages[target_end_index - 1] = replace(
            target_message,
            parts=[replace(target_message.parts[0], content=normalized_content)],
        )

    async def regenerate_from_user_message(
        self,
        *,
        user_id: int,
        conversation_id: str,
        pk: int,
        obj: AIChatRegenerateParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        根据最后一条用户消息重发 AI 回复

        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param pk: 消息主键
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        protocol_adapter = get_chat_protocol_adapter()
        run_context = protocol_adapter.build_run_context(
            conversation_id=obj.conversation_id,
            forwarded_props=obj.forwarded_props,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )
        forwarded_props = run_context.forwarded_props
        session = None
        try:
            async with async_db_session() as db:
                state = await ai_conversation_service.get_chat_state(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    must_exist=True,
                    require_messages=True,
                )
                self._get_regenerate_target(state=state, pk=pk)
                session, agent = await open_chat_session(
                    db=db,
                    forwarded_props=forwarded_props,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )

            async with async_db_session.begin() as db:
                conversation = await ai_conversation_service.get_owned_conversation(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    for_update=True,
                )
                if conversation is None:
                    raise errors.NotFoundError(msg='对话不存在')
                await ai_conversation_service.ensure_idle(db=db, conversation_id=conversation_id)
                state = await ai_conversation_service.get_chat_state(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    must_exist=True,
                    require_messages=True,
                )
                target_index, target_end_index = self._get_regenerate_target(state=state, pk=pk)
                await self._update_last_user_message_content(
                    db=db,
                    state=state,
                    row_index=target_index,
                    target_end_index=target_end_index,
                    content=obj.content,
                )

                replace_start_index, replace_end_index, insert_before_index = self._get_reply_segment_indexes(
                    message_rows=state.message_rows,
                    model_messages=state.model_messages,
                    reply_start_index=target_index + 1,
                    row_model_message_ranges=state.row_model_message_ranges,
                )
                message_index = await ai_message_dao.get_next_message_index(db, conversation_id)
                assistant_placeholder = await ai_message_dao.create(
                    db,
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': message_index,
                        'role': 'assistant',
                        'status': AIMessageStatus.pending,
                        'model_messages': [],
                    },
                )
                await ai_conversation_dao.update(
                    db,
                    conversation.id,
                    UpdateAIConversationParam(
                        conversation_id=conversation.conversation_id,
                        title=conversation.title,
                        provider_id=forwarded_props.provider_id,
                        model_id=forwarded_props.model_id,
                        user_id=conversation.user_id,
                        pinned_time=conversation.pinned_time,
                    ),
                )
                persistence = RegenerationPersistenceContext(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    forwarded_props=forwarded_props,
                    assistant_message_id=assistant_placeholder.id,
                    insert_before_index=insert_before_index,
                    replace_start_index=replace_start_index,
                    replace_end_index=replace_end_index,
                )
                message_history = state.model_messages[:target_end_index]

                async def on_complete(result: AgentRunResult[Any]) -> None:
                    async with async_db_session.begin() as callback_db:
                        await persist_regeneration(
                            db=callback_db,
                            persistence=persistence,
                            messages=extract_assistant_run_messages(result),
                        )

                async def on_run_error(message: str, messages: list[ModelRequest | ModelResponse]) -> None:
                    await persist_terminal_regeneration(
                        persistence=persistence,
                        messages=extract_assistant_messages(messages),
                        status=AIMessageStatus.error,
                        reason=message,
                    )

                async def on_interrupted(messages: list[ModelRequest | ModelResponse]) -> None:
                    await persist_terminal_regeneration(
                        persistence=persistence,
                        messages=extract_assistant_messages(messages),
                        status=AIMessageStatus.interrupted,
                    )

                response = await session.stream(
                    user_id=user_id,
                    agent=agent,
                    run_context=run_context,
                    protocol_adapter=protocol_adapter,
                    accept=accept,
                    message_history=message_history,
                    persistence=persistence,
                    on_complete=on_complete,
                    on_run_error=on_run_error,
                    on_interrupted=on_interrupted,
                )
            activate_run(conversation_id)
        except BaseException:
            # 屏蔽取消：任务取消时仍完成客户端关闭，避免连接泄漏
            with anyio.CancelScope(shield=True):
                try:
                    await abort_prepared_run(conversation_id)
                except Exception as abort_exc:
                    log.warning(f'释放聊天任务租约失败: {abort_exc}')
                if session is not None:
                    try:
                        await session.aclose()
                    except Exception as exc:
                        log.warning(f'关闭模型供应商客户端失败: {exc}')
            raise
        return response

    @staticmethod
    async def clear(
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
    ) -> int:
        """
        清空对话消息

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :return:
        """
        await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            for_update=True,
        )
        await ai_conversation_service.ensure_idle(db=db, conversation_id=conversation_id)
        return await ai_message_dao.delete(db, conversation_id)

    async def delete(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        pk: int,
    ) -> int:
        """
        删除指定消息

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param pk: 消息主键
        :return:
        """
        await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            for_update=True,
        )
        await ai_conversation_service.ensure_idle(db=db, conversation_id=conversation_id)
        message_rows = list(await ai_message_dao.get_all_by_message_index(db, conversation_id))
        target_message_index = self._get_message_row_index(message_rows=message_rows, pk=pk)
        model_messages, row_model_message_ranges = expand_message_rows(message_rows)
        if not self._is_user_message_row(
            model_messages=model_messages,
            row_index=target_message_index,
            row_model_message_ranges=row_model_message_ranges,
        ):
            target_row_messages = get_row_model_messages(
                model_messages=model_messages,
                row_message_ranges=row_model_message_ranges,
                row_index=target_message_index,
            )
            if not any(isinstance(message, ModelResponse) for message in target_row_messages):
                raise errors.RequestError(msg='仅支持删除用户消息或 AI 回复')
        count = await ai_message_dao.delete_message(db, pk)
        return count


ai_message_service: AIMessageService = AIMessageService()
