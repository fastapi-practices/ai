from pydantic_ai import ModelMessage, ModelRequest, UserPromptPart
from pydantic_core import to_jsonable_python
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.chat.runner import is_user_prompt_message, open_chat_session
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import CompletionPersistenceContext
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.protocol.registry import get_chat_protocol_adapter
from backend.plugin.ai.schema.chat import AIChatCompletionParam
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.service.conversation_service import ai_conversation_service
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title
from backend.plugin.ai.utils.message_storage import build_chat_message_record


def _get_current_user_prompt_part(*, messages: list[ModelMessage]) -> UserPromptPart:
    """
    获取当前轮用户输入部分

    :param messages: 模型消息列表
    :return:
    """
    if len(messages) != 1:
        raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')
    current_message = messages[-1]
    if not isinstance(current_message, ModelRequest) or not is_user_prompt_message(message=current_message):
        raise errors.RequestError(msg='最后一条消息必须是用户消息')
    first_part = current_message.parts[0]
    if not isinstance(first_part, UserPromptPart):
        raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')
    return first_part


def _parse_user_prompt(*, first_part: UserPromptPart) -> tuple[str, bool]:
    """
    解析用户输入文本和二进制输入状态

    :param first_part: 用户输入部分
    :return:
    """
    content_items = [first_part.content] if isinstance(first_part.content, str) else list(first_part.content)
    prompt_parts = [item for item in content_items if isinstance(item, str)]
    has_binary_input = len(prompt_parts) != len(content_items)
    prompt = ' '.join(' '.join(part.split()) for part in prompt_parts if part.split())
    return prompt, has_binary_input


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

        first_part = _get_current_user_prompt_part(messages=current_messages)
        prompt, has_binary_input = _parse_user_prompt(first_part=first_part)
        if not prompt and not has_binary_input:
            raise errors.RequestError(msg='当前轮用户消息不能为空')
        run_context = protocol_adapter.build_run_context(
            conversation_id=obj.conversation_id,
            forwarded_props=obj.forwarded_props,
        )
        conversation_id = run_context.conversation_id
        forwarded_props = run_context.forwarded_props

        agent_session = None
        try:
            async with async_db_session() as db:
                agent_session, agent = await open_chat_session(
                    db=db,
                    forwarded_props=forwarded_props,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            current_messages = protocol_adapter.sanitize_input_messages(
                agent=agent,
                run_context=run_context,
                messages=current_messages,
            )
            first_part = _get_current_user_prompt_part(messages=current_messages)
            prompt, has_binary_input = _parse_user_prompt(first_part=first_part)
            if not prompt and not has_binary_input:
                raise errors.RequestError(msg='当前轮用户消息不能为空')
            payload_messages = to_jsonable_python(current_messages, by_alias=True)
            assert isinstance(payload_messages, list)
            user_message_record = build_chat_message_record(role='user', model_messages=payload_messages)
            assistant_message_id: int | None = None
            async with async_db_session.begin() as session:
                conversation = await ai_conversation_service.get_owned_conversation(
                    db=session,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    must_exist=False,
                    for_update=True,
                )
                if conversation:
                    if await ai_message_dao.has_pending(session, conversation_id):
                        raise errors.RequestError(msg='当前对话正在生成，请稍后再试')
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
                await ai_message_dao.create(
                    session,
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': next_message_index,
                        'status': AIMessageStatus.success,
                        **user_message_record,
                    },
                )
                assistant_message = await ai_message_dao.create(
                    session,
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': next_message_index + 1,
                        'role': 'assistant',
                        'status': AIMessageStatus.pending,
                        'model_messages': [],
                    },
                )
                assistant_message_id = assistant_message.id

            async with async_db_session() as db:
                state = await ai_conversation_service.get_chat_state(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    must_exist=True,
                    require_messages=True,
                )
        except Exception:
            if agent_session is not None:
                await agent_session.aclose()
            raise
        message_history = state.model_messages[state.context_start_index :]
        persistence = CompletionPersistenceContext(
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            title=state.conversation.title if state.conversation else prompt,
            assistant_message_id=assistant_message_id,
        )

        return agent_session.stream(
            user_id=user_id,
            agent=agent,
            run_context=run_context,
            protocol_adapter=protocol_adapter,
            accept=accept,
            message_history=message_history,
            persistence=persistence,
        )


ai_chat_service: ChatService = ChatService()
