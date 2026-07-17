from copy import deepcopy
from dataclasses import replace
from datetime import datetime
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
from backend.plugin.ai.chat.session import AgentSession
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import (
    ChatConversationState,
    ChatRunContext,
    RegenerationPersistenceContext,
)
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.model import AIMessage
from backend.plugin.ai.protocol.base import ChatAgent, ChatProtocolAdapter
from backend.plugin.ai.protocol.registry import get_chat_protocol_adapter
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam, AIChatRegenerateParam
from backend.plugin.ai.schema.conversation import UpdateAIConversationParam
from backend.plugin.ai.schema.message import UpdateAIMessageParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.message_storage import (
    expand_message_rows,
    get_message_row_model_message_payloads,
    get_row_model_messages,
)


class AIMessageService:
    """AI 消息服务"""

    @staticmethod
    def _get_row_model_message_ranges(
        *,
        message_rows: list[AIMessage],
        model_messages: list[ModelRequest | ModelResponse],
        row_model_message_ranges: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """
        获取消息行到模型消息的范围映射

        :param message_rows: 消息行列表
        :param model_messages: 模型消息列表
        :param row_model_message_ranges: 已存在的范围映射
        :return:
        """
        if row_model_message_ranges is not None:
            return row_model_message_ranges
        if len(message_rows) == len(model_messages):
            return [(index, index + 1) for index in range(len(message_rows))]
        _, ranges = expand_message_rows(message_rows)
        return ranges

    def _get_row_messages(
        self,
        *,
        message_rows: list[AIMessage],
        model_messages: list[ModelRequest | ModelResponse],
        row_index: int,
        row_model_message_ranges: list[tuple[int, int]] | None = None,
    ) -> list[ModelRequest | ModelResponse]:
        """
        获取消息行对应的模型消息

        :param message_rows: 消息行列表
        :param model_messages: 模型消息列表
        :param row_index: 消息行索引
        :param row_model_message_ranges: 行到模型消息范围映射
        :return:
        """
        ranges = self._get_row_model_message_ranges(
            message_rows=message_rows,
            model_messages=model_messages,
            row_model_message_ranges=row_model_message_ranges,
        )
        return get_row_model_messages(
            model_messages=model_messages,
            row_message_ranges=ranges,
            row_index=row_index,
        )

    def _is_user_message_row(
        self,
        *,
        message_rows: list[AIMessage],
        model_messages: list[ModelRequest | ModelResponse],
        row_index: int,
        row_model_message_ranges: list[tuple[int, int]] | None = None,
    ) -> bool:
        """
        判断是否为用户消息行

        :param message_rows: 消息行列表
        :param model_messages: 模型消息列表
        :param row_index: 消息行索引
        :param row_model_message_ranges: 行到模型消息范围映射
        :return:
        """
        row_messages = self._get_row_messages(
            message_rows=message_rows,
            model_messages=model_messages,
            row_index=row_index,
            row_model_message_ranges=row_model_message_ranges,
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
            raise errors.NotFoundError(msg='消息不存在')
        return message_row_index

    def _get_reply_segment_indexes(
        self,
        *,
        message_rows: list[AIMessage],
        model_messages: list[ModelRequest | ModelResponse],
        reply_start_index: int,
        row_model_message_ranges: list[tuple[int, int]] | None = None,
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
            message_rows=message_rows,
            model_messages=model_messages,
            row_index=reply_start_index,
            row_model_message_ranges=row_model_message_ranges,
        ):
            return None, None, message_rows[reply_start_index].message_index
        reply_end_index = reply_start_index
        for index in range(reply_start_index + 1, len(message_rows)):
            if self._is_user_message_row(
                message_rows=message_rows,
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
    async def _prepare_regenerate_context(
        *,
        user_id: int,
        conversation_id: str,
        obj: AIChatRegenerateParam,
    ) -> tuple[
        ChatRunContext,
        AIChatForwardedPropsParam,
        AgentSession,
        ChatAgent,
        ChatConversationState,
        ChatProtocolAdapter,
    ]:
        """
        预加载重生成所需上下文

        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param obj: 请求体
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
                session, agent = await open_chat_session(
                    db=db,
                    forwarded_props=forwarded_props,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
        except BaseException:
            # shield：任务取消时仍完成客户端关闭，避免连接泄漏
            with anyio.CancelScope(shield=True):
                if session is not None:
                    try:
                        await session.aclose()
                    except Exception as exc:
                        log.warning(f'关闭模型供应商客户端失败: {exc}')
            raise
        return run_context, forwarded_props, session, agent, state, protocol_adapter

    @staticmethod
    def _build_message_versions(*, message_rows: list[AIMessage]) -> tuple[tuple[int, datetime | None], ...]:
        """构建可检测原地修改的消息快照版本"""
        return tuple((row.id, row.updated_time) for row in message_rows)

    async def _create_regeneration_placeholder(
        self,
        *,
        db: AsyncSession,
        persistence: RegenerationPersistenceContext,
    ) -> RegenerationPersistenceContext:
        """
        创建重生成占位消息并校验预加载快照未发生变化

        :param db: 数据库会话
        :param persistence: 重生成持久化上下文
        :return:
        """
        conversation = await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=persistence.conversation_id,
            user_id=persistence.user_id,
            for_update=True,
        )
        assert conversation is not None
        await ai_conversation_service.ensure_idle(
            db=db,
            conversation_id=persistence.conversation_id,
        )
        message_rows = list(await ai_message_dao.get_all_by_message_index(db, persistence.conversation_id))
        message_versions = self._build_message_versions(message_rows=message_rows)
        context_changed = (
            conversation.context_start_message_id != persistence.expected_context_start_message_id
            or conversation.context_cleared_time != persistence.expected_context_cleared_time
        )
        if (
            persistence.expected_message_versions and message_versions != persistence.expected_message_versions
        ) or context_changed:
            raise errors.ConflictError(msg='对话消息已发生变化，请重试')

        message_index = await ai_message_dao.get_next_message_index(db, persistence.conversation_id)
        assistant_placeholder = await ai_message_dao.create(
            db,
            {
                'conversation_id': persistence.conversation_id,
                'provider_id': persistence.forwarded_props.provider_id,
                'model_id': persistence.forwarded_props.model_id,
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
                provider_id=persistence.forwarded_props.provider_id,
                model_id=persistence.forwarded_props.model_id,
                user_id=conversation.user_id,
                pinned_time=conversation.pinned_time,
                context_start_message_id=conversation.context_start_message_id,
                context_cleared_time=conversation.context_cleared_time,
            ),
        )
        return replace(persistence, assistant_message_id=assistant_placeholder.id)

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
        根据用户消息重生成 AI 回复

        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param pk: 消息主键
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_context, forwarded_props, session, agent, state, protocol_adapter = await self._prepare_regenerate_context(
            user_id=user_id,
            conversation_id=conversation_id,
            obj=obj,
        )
        try:
            conversation = state.conversation
            assert conversation is not None
            row_model_message_ranges = getattr(state, 'row_model_message_ranges', None)
            target_index = self._get_message_row_index(message_rows=state.message_rows, pk=pk)
            target_messages = self._get_row_messages(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                row_index=target_index,
                row_model_message_ranges=row_model_message_ranges,
            )
            if len(target_messages) != 1 or not is_user_prompt_message(message=target_messages[0]):
                raise errors.RequestError(msg='仅支持根据用户消息重生成')
            target_start_index, target_end_index = self._get_row_model_message_ranges(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                row_model_message_ranges=row_model_message_ranges,
            )[target_index]
            if target_start_index < state.context_start_index:
                raise errors.RequestError(msg='指定消息已不在当前上下文中')

            reply_start_index = target_index + 1
            message_history = state.model_messages[state.context_start_index : target_end_index]
            expected_message_versions = self._build_message_versions(message_rows=state.message_rows)
            replace_start_index, replace_end_index, insert_before_index = self._get_reply_segment_indexes(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                reply_start_index=reply_start_index,
                row_model_message_ranges=row_model_message_ranges,
            )
            if replace_start_index is not None:
                persistence = RegenerationPersistenceContext(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    forwarded_props=forwarded_props,
                    expected_message_versions=expected_message_versions,
                    expected_context_start_message_id=conversation.context_start_message_id,
                    expected_context_cleared_time=conversation.context_cleared_time,
                    replace_start_index=replace_start_index,
                    replace_end_index=replace_end_index,
                )
            elif insert_before_index is not None:
                persistence = RegenerationPersistenceContext(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    forwarded_props=forwarded_props,
                    expected_message_versions=expected_message_versions,
                    expected_context_start_message_id=conversation.context_start_message_id,
                    expected_context_cleared_time=conversation.context_cleared_time,
                    insert_before_index=insert_before_index,
                )
            else:
                persistence = RegenerationPersistenceContext(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    forwarded_props=forwarded_props,
                    expected_message_versions=expected_message_versions,
                    expected_context_start_message_id=conversation.context_start_message_id,
                    expected_context_cleared_time=conversation.context_cleared_time,
                )

            async with async_db_session.begin() as db:
                persistence = await self._create_regeneration_placeholder(db=db, persistence=persistence)

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

                response = session.stream(
                    user_id=user_id,
                    agent=agent,
                    run_context=run_context,
                    protocol_adapter=protocol_adapter,
                    accept=accept,
                    message_history=message_history,
                    on_complete=on_complete,
                    on_run_error=on_run_error,
                    on_interrupted=on_interrupted,
                )
        except BaseException:
            # shield：任务取消时仍完成客户端关闭，避免连接泄漏
            with anyio.CancelScope(shield=True):
                try:
                    await session.aclose()
                except Exception as exc:
                    log.warning(f'关闭模型供应商客户端失败: {exc}')
            raise
        return response

    async def regenerate_from_response_message(
        self,
        *,
        user_id: int,
        conversation_id: str,
        pk: int,
        obj: AIChatRegenerateParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        根据 AI 回复重生成

        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param pk: 消息主键
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_context, forwarded_props, session, agent, state, protocol_adapter = await self._prepare_regenerate_context(
            user_id=user_id,
            conversation_id=conversation_id,
            obj=obj,
        )
        try:
            conversation = state.conversation
            assert conversation is not None
            row_model_message_ranges = getattr(state, 'row_model_message_ranges', None)
            target_index = self._get_message_row_index(message_rows=state.message_rows, pk=pk)
            target_messages = self._get_row_messages(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                row_index=target_index,
                row_model_message_ranges=row_model_message_ranges,
            )
            if not any(isinstance(message, ModelResponse) for message in target_messages):
                raise errors.RequestError(msg='仅支持根据 AI 回复重生成')
            target_start_index, _ = self._get_row_model_message_ranges(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                row_model_message_ranges=row_model_message_ranges,
            )[target_index]
            if target_start_index < state.context_start_index:
                raise errors.RequestError(msg='指定消息已不在当前上下文中')

            row_ranges = self._get_row_model_message_ranges(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                row_model_message_ranges=row_model_message_ranges,
            )
            user_message_index = target_index - 1
            target_row = state.message_rows[target_index]
            has_adjacent_user_message = (
                user_message_index >= 0
                and state.message_rows[user_message_index].message_index == target_row.message_index - 1
                and row_ranges[user_message_index][0] >= state.context_start_index
                and self._is_user_message_row(
                    message_rows=state.message_rows,
                    model_messages=state.model_messages,
                    row_index=user_message_index,
                    row_model_message_ranges=row_model_message_ranges,
                )
            )
            if not has_adjacent_user_message:
                raise errors.RequestError(msg='未找到对应的用户消息')

            _, user_message_end_index = row_ranges[user_message_index]
            message_history = state.model_messages[state.context_start_index : user_message_end_index]
            replace_start_index, replace_end_index, _ = self._get_reply_segment_indexes(
                message_rows=state.message_rows,
                model_messages=state.model_messages,
                reply_start_index=user_message_index + 1,
                row_model_message_ranges=row_model_message_ranges,
            )
            if replace_start_index is None:
                raise errors.RequestError(msg='未找到对应的 AI 回复段')
            persistence = RegenerationPersistenceContext(
                conversation_id=conversation_id,
                user_id=user_id,
                forwarded_props=forwarded_props,
                expected_message_versions=self._build_message_versions(message_rows=state.message_rows),
                expected_context_start_message_id=conversation.context_start_message_id,
                expected_context_cleared_time=conversation.context_cleared_time,
                replace_start_index=replace_start_index,
                replace_end_index=replace_end_index,
            )

            async with async_db_session.begin() as db:
                persistence = await self._create_regeneration_placeholder(db=db, persistence=persistence)

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

                response = session.stream(
                    user_id=user_id,
                    agent=agent,
                    run_context=run_context,
                    protocol_adapter=protocol_adapter,
                    accept=accept,
                    message_history=message_history,
                    on_complete=on_complete,
                    on_run_error=on_run_error,
                    on_interrupted=on_interrupted,
                )
        except BaseException:
            # shield：任务取消时仍完成客户端关闭，避免连接泄漏
            with anyio.CancelScope(shield=True):
                try:
                    await session.aclose()
                except Exception as exc:
                    log.warning(f'关闭模型供应商客户端失败: {exc}')
            raise
        return response

    async def update(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        pk: int,
        obj: UpdateAIMessageParam,
    ) -> int:
        """
        编辑保存指定消息

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param pk: 消息主键
        :param obj: 更新参数
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
        model_messages, row_model_message_ranges = expand_message_rows(message_rows)
        message_row_index = self._get_message_row_index(message_rows=message_rows, pk=pk)
        target_messages = self._get_row_messages(
            message_rows=message_rows,
            model_messages=model_messages,
            row_index=message_row_index,
            row_model_message_ranges=row_model_message_ranges,
        )
        target_message = target_messages[0] if target_messages else None
        if len(target_messages) != 1 or not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not isinstance(target_message.parts[0].content, str):
            raise errors.RequestError(msg='当前消息暂不支持直接编辑')

        content = ' '.join(obj.content.split())
        if not content:
            raise errors.RequestError(msg='消息内容不能为空')
        model_messages_payload = deepcopy(get_message_row_model_message_payloads(message_rows[message_row_index]))
        model_payload = deepcopy(model_messages_payload[0])
        model_payload['parts'][0]['content'] = content
        model_messages_payload[0] = model_payload
        return await ai_message_dao.update(db, pk, {'model_messages': model_messages_payload})

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
        conversation = await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            for_update=True,
        )
        await ai_conversation_service.ensure_idle(db=db, conversation_id=conversation_id)
        await ai_conversation_dao.update(
            db,
            conversation.id,
            UpdateAIConversationParam(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                provider_id=conversation.provider_id,
                model_id=conversation.model_id,
                user_id=conversation.user_id,
                pinned_time=conversation.pinned_time,
                context_start_message_id=None,
                context_cleared_time=None,
            ),
        )
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
        conversation = await ai_conversation_service.get_owned_conversation(
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
            message_rows=message_rows,
            model_messages=model_messages,
            row_index=target_message_index,
            row_model_message_ranges=row_model_message_ranges,
        ):
            target_row_messages = self._get_row_messages(
                message_rows=message_rows,
                model_messages=model_messages,
                row_index=target_message_index,
                row_model_message_ranges=row_model_message_ranges,
            )
            if not any(isinstance(message, ModelResponse) for message in target_row_messages):
                raise errors.RequestError(msg='仅支持删除用户消息或 AI 回复')
        target_row = message_rows[target_message_index]

        count = await ai_message_dao.delete_message(db, pk)

        context_start_message_id = conversation.context_start_message_id
        if context_start_message_id == pk:
            previous_rows = [row for row in message_rows if row.message_index < target_row.message_index]
            context_start_message_id = previous_rows[-1].id if previous_rows else None
            await ai_conversation_dao.update(
                db,
                conversation.id,
                UpdateAIConversationParam(
                    conversation_id=conversation.conversation_id,
                    title=conversation.title,
                    provider_id=conversation.provider_id,
                    model_id=conversation.model_id,
                    user_id=conversation.user_id,
                    pinned_time=conversation.pinned_time,
                    context_start_message_id=context_start_message_id,
                    context_cleared_time=conversation.context_cleared_time if context_start_message_id else None,
                ),
            )
        return count


ai_message_service: AIMessageService = AIMessageService()
