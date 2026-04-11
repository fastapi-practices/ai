from copy import deepcopy
from functools import partial

from ag_ui.core import RunAgentInput
from pydantic_ai import Agent, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.plugin.ai.chat_runtime import (
    build_chat_agent,
    is_user_prompt_message,
    persist_completion_result,
    prepare_run_input,
    stream_response,
)
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import ChatCompletionPersistence, ChatConversationState
from backend.plugin.ai.model import AIMessage
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam, AIChatRegenerateParam
from backend.plugin.ai.schema.message import UpdateAIMessageParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.conversation_control import build_update_ai_conversation_param


class AIMessageService:
    """AI 消息服务"""

    @staticmethod
    def _get_message_row_index(*, message_rows: list[AIMessage], message_id: int) -> int:
        """
        获取消息行索引

        :param message_rows: 消息行列表
        :param message_id: 消息 ID
        :return:
        """
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='消息不存在')
        return message_row_index

    @staticmethod
    def _find_next_user_message_index(
        *,
        model_messages: list[ModelRequest | ModelResponse],
        start_index: int,
    ) -> int | None:
        """
        获取后续首条用户消息索引

        :param model_messages: 模型消息列表
        :param start_index: 起始索引
        :return:
        """
        return next(
            (
                index
                for index in range(start_index, len(model_messages))
                if is_user_prompt_message(message=model_messages[index])
            ),
            None,
        )

    @staticmethod
    async def _prepare_regenerate_context(
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        obj: AIChatRegenerateParam,
    ) -> tuple[RunAgentInput, AIChatForwardedPropsParam, Agent, ChatConversationState]:
        """
        预加载重生成所需上下文

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param obj: 请求体
        :return:
        """
        run_input = prepare_run_input(
            conversation_id=obj.conversation_id,
            forwarded_props=obj.forwarded_props,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )
        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        agent = await build_chat_agent(db=db, forwarded_props=forwarded_props)
        state = await ai_conversation_service.get_chat_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )
        return run_input, forwarded_props, agent, state

    @staticmethod
    async def clear(
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
    ) -> int:
        """
        清空对话消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        await ai_conversation_dao.update(
            db,
            conversation.id,
            build_update_ai_conversation_param(
                conversation=conversation,
                context_start_message_id=None,
                context_cleared_time=None,
            ),
        )
        return await ai_message_dao.delete(db, conversation_id)

    async def update(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
        obj: UpdateAIMessageParam,
    ) -> int:
        """
        编辑保存指定消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :param obj: 更新参数
        :return:
        """
        await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        message_row_index = self._get_message_row_index(message_rows=message_rows, message_id=message_id)
        target_message = model_messages[message_row_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not isinstance(target_message.parts[0].content, str):
            raise errors.RequestError(msg='当前消息暂不支持直接编辑')

        content = ' '.join(obj.content.split())
        if not content:
            raise errors.RequestError(msg='消息内容不能为空')
        payload = deepcopy(message_rows[message_row_index].message)
        payload['parts'][0]['content'] = content
        return await ai_message_dao.update(db, message_id, {'message': payload})

    async def regenerate_from_user_message(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        message_id: int,
        obj: AIChatRegenerateParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        根据用户消息重生成 AI 回复

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param message_id: 消息 ID
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_input, forwarded_props, agent, state = await self._prepare_regenerate_context(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            obj=obj,
        )
        target_index = self._get_message_row_index(message_rows=state.message_rows, message_id=message_id)
        target_message = state.model_messages[target_index]
        if not isinstance(target_message, ModelRequest) or not is_user_prompt_message(message=target_message):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        reply_start_index = target_index + 1
        message_history = state.model_messages[state.context_start_index : target_index + 1]
        has_existing_reply = reply_start_index < len(state.model_messages) and not is_user_prompt_message(
            message=state.model_messages[reply_start_index]
        )
        if has_existing_reply:
            next_user_message_index = self._find_next_user_message_index(
                model_messages=state.model_messages,
                start_index=reply_start_index + 1,
            )
            reply_end_index = (
                next_user_message_index - 1 if next_user_message_index is not None else len(state.model_messages) - 1
            )
            persistence = ChatCompletionPersistence(
                conversation_id=conversation_id,
                user_id=user_id,
                forwarded_props=forwarded_props,
                conversation=state.conversation,
                title=state.conversation.title,
                replace_message_row_ids=[row.id for row in state.message_rows[reply_start_index : reply_end_index + 1]],
                replace_start_message_index=reply_start_index,
                replace_end_message_index=reply_end_index,
                insert_before_message_index=None,
                base_message_index=reply_start_index,
                result_offset=len(message_history),
            )
        else:
            persistence = ChatCompletionPersistence(
                conversation_id=conversation_id,
                user_id=user_id,
                forwarded_props=forwarded_props,
                conversation=state.conversation,
                title=state.conversation.title,
                replace_message_row_ids=None,
                replace_start_message_index=None,
                replace_end_message_index=None,
                insert_before_message_index=reply_start_index if reply_start_index < len(state.message_rows) else None,
                base_message_index=reply_start_index,
                result_offset=len(message_history),
            )

        return stream_response(
            db=db,
            user_id=user_id,
            agent=agent,
            run_input=run_input,
            accept=accept,
            message_history=message_history,
            on_complete=partial(persist_completion_result, db=db, persistence=persistence),
            persistence=persistence,
        )

    async def regenerate_from_response_message(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        message_id: int,
        obj: AIChatRegenerateParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        根据 AI 回复重生成

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param message_id: 消息 ID
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_input, forwarded_props, agent, state = await self._prepare_regenerate_context(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            obj=obj,
        )
        target_index = self._get_message_row_index(message_rows=state.message_rows, message_id=message_id)
        if not isinstance(state.model_messages[target_index], ModelResponse):
            raise errors.RequestError(msg='仅支持根据 AI 回复重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        user_message_index = next(
            (
                index
                for index in range(target_index - 1, state.context_start_index - 1, -1)
                if is_user_prompt_message(message=state.model_messages[index])
            ),
            None,
        )
        if user_message_index is None:
            raise errors.RequestError(msg='未找到对应的用户消息')

        reply_start_index = user_message_index + 1
        next_user_message_index = self._find_next_user_message_index(
            model_messages=state.model_messages,
            start_index=reply_start_index + 1,
        )
        reply_end_index = (
            next_user_message_index - 1 if next_user_message_index is not None else len(state.model_messages) - 1
        )
        message_history = state.model_messages[state.context_start_index : user_message_index + 1]
        persistence = ChatCompletionPersistence(
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title,
            replace_message_row_ids=[row.id for row in state.message_rows[reply_start_index : reply_end_index + 1]],
            replace_start_message_index=reply_start_index,
            replace_end_message_index=reply_end_index,
            insert_before_message_index=None,
            base_message_index=reply_start_index,
            result_offset=len(message_history),
        )

        return stream_response(
            db=db,
            user_id=user_id,
            agent=agent,
            run_input=run_input,
            accept=accept,
            message_history=message_history,
            on_complete=partial(persist_completion_result, db=db, persistence=persistence),
            persistence=persistence,
        )

    async def delete(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
    ) -> int:
        """
        删除指定消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :return:
        """
        conversation = await ai_conversation_service.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        target_message_index = self._get_message_row_index(message_rows=message_rows, message_id=message_id)

        remaining_message_rows = [row for row in message_rows if row.id != message_id]
        if not remaining_message_rows:
            await ai_message_dao.delete(db, conversation_id)
            return await ai_conversation_dao.delete(db, conversation_id, user_id)

        await ai_message_dao.delete_message(db, message_id)
        for index, row in enumerate(remaining_message_rows):
            if row.message_index != index:
                await ai_message_dao.update(db, row.id, {'message_index': index})

        context_start_message_id = conversation.context_start_message_id
        if context_start_message_id == message_id:
            previous_rows = message_rows[:target_message_index]
            context_start_message_id = previous_rows[-1].id if previous_rows else None
        return await ai_conversation_dao.update(
            db,
            conversation.id,
            build_update_ai_conversation_param(
                conversation=conversation,
                provider_id=remaining_message_rows[-1].provider_id,
                model_id=remaining_message_rows[-1].model_id,
                context_start_message_id=context_start_message_id,
                context_cleared_time=conversation.context_cleared_time if context_start_message_id else None,
            ),
        )


ai_message_service: AIMessageService = AIMessageService()
