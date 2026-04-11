from functools import partial

from pydantic_ai import ModelMessage, ModelRequest, UserPromptPart
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.chat_runtime import (
    build_chat_agent,
    is_user_prompt_message,
    persist_completion_result,
    prepare_run_input,
    stream_response,
)
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import ChatCompletionPersistence
from backend.plugin.ai.protocol.ag_ui.request_decoder import decode_input_messages
from backend.plugin.ai.schema.chat import AIChatCompletionParam, AIChatForwardedPropsParam
from backend.plugin.ai.schema.conversation import CreateAIConversationParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.conversation_control import (
    build_update_ai_conversation_param,
    normalize_generated_conversation_title,
)


class ChatService:
    """聊天服务"""

    @staticmethod
    def _extract_prompt(*, current_message: ModelRequest) -> str:
        """
        提取当前轮用户输入文本

        :param current_message: 当前轮消息
        :return:
        """
        if not current_message.parts:
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')
        first_part = current_message.parts[0]
        if not isinstance(first_part, UserPromptPart):
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')

        prompt_parts: list[str] = []
        has_binary_input = False
        if isinstance(first_part.content, str):
            prompt_parts.append(first_part.content)
        else:
            for item in first_part.content:
                if isinstance(item, str):
                    prompt_parts.append(item)
                else:
                    has_binary_input = True

        prompt = ' '.join(part.strip() for part in prompt_parts if part.strip()).strip()
        if not prompt and not has_binary_input:
            raise errors.RequestError(msg='当前轮用户消息不能为空')
        return prompt

    @staticmethod
    def _get_latest_user_message(*, messages: list[ModelMessage]) -> ModelRequest:
        """
        获取当前轮最后一条用户消息

        :param messages: 当前轮模型消息列表
        :return:
        """
        if not messages:
            raise errors.RequestError(msg='当前轮消息不能为空')
        current_message = messages[-1]
        if not isinstance(current_message, ModelRequest) or not is_user_prompt_message(message=current_message):
            raise errors.RequestError(msg='最后一条消息必须是用户消息')
        return current_message

    async def create_completion(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        obj: AIChatCompletionParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        创建流式对话

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        try:
            current_messages = decode_input_messages(messages=obj.messages)
        except Exception as e:
            log.warning(f'聊天消息加载失败: {e}')
            raise errors.RequestError(msg='聊天消息格式非法') from e
        current_message = self._get_latest_user_message(messages=current_messages)
        prompt = self._extract_prompt(current_message=current_message)
        run_input = prepare_run_input(
            conversation_id=obj.conversation_id,
            forwarded_props=obj.forwarded_props,
        )
        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        agent = await build_chat_agent(db=db, forwarded_props=forwarded_props)
        conversation_id = run_input.thread_id
        payload_messages = to_jsonable_python(current_messages, by_alias=True)
        assert isinstance(payload_messages, list)
        # 使用独立事务先提交用户输入，避免流式阶段异常导致整段会话回滚
        async with async_db_session.begin() as session:
            conversation = await ai_conversation_service.get_owned_conversation(
                db=session,
                conversation_id=conversation_id,
                user_id=user_id,
                must_exist=False,
            )
            if conversation:
                await ai_conversation_dao.update(
                    session,
                    conversation.id,
                    build_update_ai_conversation_param(
                        conversation=conversation,
                        provider_id=forwarded_props.provider_id,
                        model_id=forwarded_props.model_id,
                    ),
                )
            else:
                await ai_conversation_dao.create(
                    session,
                    CreateAIConversationParam(
                        conversation_id=conversation_id,
                        title=normalize_generated_conversation_title(title=prompt),
                        provider_id=forwarded_props.provider_id,
                        model_id=forwarded_props.model_id,
                        user_id=user_id,
                    ),
                )

            message_rows = await ai_message_dao.get_all(session, conversation_id)
            await ai_message_dao.bulk_create(
                session,
                [
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': len(message_rows) + index,
                        'message': message,
                    }
                    for index, message in enumerate(payload_messages)
                ],
            )
        state = await ai_conversation_service.get_chat_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )
        message_history = state.model_messages[state.context_start_index :]
        persistence = ChatCompletionPersistence(
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title if state.conversation else prompt,
            replace_message_row_ids=None,
            replace_start_message_index=None,
            replace_end_message_index=None,
            insert_before_message_index=None,
            base_message_index=len(state.message_rows),
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


ai_chat_service: ChatService = ChatService()
