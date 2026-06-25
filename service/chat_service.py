from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_core import to_jsonable_python
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.chat.runner import is_user_prompt_message, open_chat_session
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import CompletionPersistenceContext
from backend.plugin.ai.protocol.registry import get_chat_protocol_adapter
from backend.plugin.ai.schema.chat import AIChatCompletionParam
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title


class ChatService:
    """聊天服务"""

    @staticmethod
    async def create_completion(
        *,
        user_id: int,
        obj: AIChatCompletionParam,
        accept: str | None,
    ) -> StreamingResponse:
        """
        创建流式对话

        :param user_id: 用户 ID
        :param obj: 请求体
        :param accept: Accept 请求头
        :return:
        """
        protocol_adapter = get_chat_protocol_adapter()
        try:
            current_messages = protocol_adapter.decode_input_messages(messages=obj.messages)
        except Exception as e:
            log.warning(f'聊天消息加载失败: {e}')
            raise errors.RequestError(msg='聊天消息格式非法') from e

        if not current_messages:
            raise errors.RequestError(msg='当前轮消息不能为空')
        current_message = current_messages[-1]
        if not isinstance(current_message, ModelRequest) or not is_user_prompt_message(message=current_message):
            raise errors.RequestError(msg='最后一条消息必须是用户消息')

        run_context = protocol_adapter.build_run_context(
            conversation_id=obj.conversation_id,
            forwarded_props=obj.forwarded_props,
        )
        conversation_id = run_context.conversation_id
        forwarded_props = run_context.forwarded_props
        if not current_message.parts:
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')
        first_part = current_message.parts[0]
        if not isinstance(first_part, UserPromptPart):
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')

        content_items = [first_part.content] if isinstance(first_part.content, str) else first_part.content
        prompt_parts = [item for item in content_items if isinstance(item, str)]
        has_binary_input = len(prompt_parts) != len(content_items)

        prompt = ' '.join(' '.join(part.split()) for part in prompt_parts if part.split())
        if not prompt and not has_binary_input:
            raise errors.RequestError(msg='当前轮用户消息不能为空')
        payload_messages = to_jsonable_python(current_messages, by_alias=True)

        async with async_db_session.begin() as session:
            conversation = await ai_conversation_service.get_owned_conversation(
                db=session,
                conversation_id=conversation_id,
                user_id=user_id,
                must_exist=False,
                for_update=True,
            )
            if conversation:
                await ai_conversation_dao.update(
                    session,
                    conversation.id,
                    UpdateAIConversationParam(
                        conversation_id=conversation.conversation_id,
                        title=conversation.title,
                        provider_id=forwarded_props.provider_id,
                        model_id=forwarded_props.model_id,
                        user_id=conversation.user_id,
                        pinned_time=conversation.pinned_time,
                        context_start_message_id=conversation.context_start_message_id,
                        context_cleared_time=conversation.context_cleared_time,
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

            next_message_index = await ai_message_dao.get_next_message_index(session, conversation_id)
            await ai_message_dao.bulk_create(
                session,
                [
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': next_message_index + index,
                        'message': message,
                    }
                    for index, message in enumerate(payload_messages)
                ],
            )

        async with async_db_session() as db:
            session, agent = await open_chat_session(db=db, forwarded_props=forwarded_props)
            state = await ai_conversation_service.get_chat_state(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                must_exist=True,
                require_messages=True,
            )
        message_history = state.model_messages[state.context_start_index :]
        persistence = CompletionPersistenceContext(
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            title=state.conversation.title if state.conversation else prompt,
            result_offset=len(message_history),
        )

        return session.stream(
            user_id=user_id,
            agent=agent,
            run_context=run_context,
            protocol_adapter=protocol_adapter,
            accept=accept,
            message_history=message_history,
            persistence=persistence,
        )


ai_chat_service: ChatService = ChatService()
