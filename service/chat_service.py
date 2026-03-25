import json

from collections.abc import AsyncGenerator
from typing import Any

from pydantic_ai import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import uuid4_str
from backend.plugin.ai.crud.crud_chat_history import ai_chat_history_dao
from backend.plugin.ai.crud.crud_chat_message import ai_chat_message_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.schema.chat import AIChatParam
from backend.plugin.ai.schema.chat_history import CreateAIChatHistoryParam, UpdateAIChatHistoryParam
from backend.plugin.ai.service.chat_history_service import ai_chat_history_service
from backend.plugin.ai.utils.chat_control import build_model_settings, chat_agent
from backend.plugin.ai.utils.message_parse import (
    build_chat_transcript,
    make_chat_message,
    parse_model_messages,
    serialize_model_messages,
    to_chat_message,
)
from backend.plugin.ai.utils.model_control import get_pydantic_model


class ChatService:
    """聊天服务类"""

    @staticmethod
    async def stream_messages(  # noqa: C901
        *,
        db: AsyncSession,
        chat: AIChatParam,
        user_id: int,
    ) -> AsyncGenerator[bytes, Any]:
        """
        流式消息

        :param db: 数据库会话
        :param chat: 聊天参数
        :param user_id: 用户 ID
        :return:
        """
        provider = await ai_provider_dao.get(db, chat.provider_id)
        if not provider:
            raise errors.NotFoundError(msg='供应商不存在')
        if not provider.status:
            raise errors.RequestError(msg='此供应商暂不可用，请更换供应商或联系系统管理员')

        model = await ai_model_dao.get_by_model_and_provider(db, chat.model_id, chat.provider_id)
        if not model:
            raise errors.NotFoundError(msg='供应商模型不存在')
        if not model.status:
            raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')

        conversation_id = chat.conversation_id or uuid4_str()
        chat_history = None
        existing_message_rows = []
        message_history = []
        next_message_index = 0
        prompt = chat.user_prompt
        should_emit_user_message = True
        if chat.conversation_id:
            chat_history = await ai_chat_history_service.get_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            existing_message_rows = list(await ai_chat_message_dao.get_all(db, conversation_id))
            if chat.edit_message_index is not None and chat.regenerate_message_index is not None:
                raise errors.RequestError(msg='编辑重发与重新生成不能同时使用')
            if chat.edit_message_index is not None:
                if prompt is None:
                    raise errors.RequestError(msg='编辑重发时用户提示词不能为空')
                chat_history, _, message_history = await ai_chat_history_service.get_editable_message(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_index=chat.edit_message_index,
                )
            elif chat.regenerate_message_index is not None:
                chat_history, prompt, message_history = await ai_chat_history_service.get_regeneratable_message(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_index=chat.regenerate_message_index,
                )
                should_emit_user_message = False
            else:
                if prompt is None:
                    raise errors.RequestError(msg='用户提示词不能为空')
                message_history = parse_model_messages([row.message for row in existing_message_rows])
            next_message_index = len(build_chat_transcript(message_history, conversation_id=conversation_id))
        else:
            if chat.edit_message_index is not None:
                raise errors.RequestError(msg='编辑重发必须指定会话 ID')
            if chat.regenerate_message_index is not None:
                raise errors.RequestError(msg='重新生成必须指定会话 ID')
            if prompt is None:
                raise errors.RequestError(msg='用户提示词不能为空')
        assert prompt is not None

        def _dump_message(model_message: ModelRequest | ModelResponse, *, message_index: int) -> bytes:
            return (
                json.dumps(
                    to_chat_message(
                        model_message,
                        message_index=message_index,
                        conversation_id=conversation_id,
                    ),
                    ensure_ascii=False,
                ).encode('utf-8')
                + b'\n'
            )

        def _build_title() -> str:
            title = chat_history.title if chat_history else ' '.join(prompt.split())
            if not title:
                return '新会话'
            if len(title) > 256:
                return title[:253] + '...'
            return title

        async def _save_history(messages: list[dict[str, object]]) -> None:
            payload = {
                'conversation_id': conversation_id,
                'title': _build_title(),
                'provider_id': chat.provider_id,
                'model_id': chat.model_id,
                'user_id': chat_history.user_id if chat_history else user_id,
                'pinned_time': chat_history.pinned_time if chat_history else None,
            }
            if chat_history:
                await ai_chat_history_dao.update(db, chat_history.id, UpdateAIChatHistoryParam(**payload))
            else:
                await ai_chat_history_dao.create(db, CreateAIChatHistoryParam(**payload))
            await ai_chat_message_dao.delete(db, conversation_id)
            if messages:
                await ai_chat_message_dao.bulk_create(
                    db,
                    [
                        {
                            'conversation_id': conversation_id,
                            'provider_id': existing_message_rows[index].provider_id
                            if index < len(message_history) and index < len(existing_message_rows)
                            else chat.provider_id,
                            'model_id': existing_message_rows[index].model_id
                            if index < len(message_history) and index < len(existing_message_rows)
                            else chat.model_id,
                            'message_index': index,
                            'message': message,
                        }
                        for index, message in enumerate(messages)
                    ],
                )

        if should_emit_user_message:
            yield json.dumps(
                make_chat_message(
                    message_index=next_message_index,
                    role='user',
                    content=prompt,
                    conversation_id=conversation_id,
                ),
                ensure_ascii=False,
            ).encode('utf-8') + b'\n'

        model_settings = build_model_settings(chat=chat, provider_type=provider.type)
        run_kwargs: dict[str, Any] = {
            'model': get_pydantic_model(
                provider_type=provider.type,
                model_name=model.model_id,
                api_key=provider.api_key,
                base_url=provider.api_host,
                model_settings=model_settings,
                provider_name=provider.name,
            )
        }
        if message_history:
            run_kwargs['message_history'] = message_history

        try:
            async with chat_agent.run_stream(
                prompt,
                **run_kwargs,
            ) as result:
                async for text in result.stream_output(debounce_by=0.01):
                    response_message = ModelResponse(
                        parts=[TextPart(text)],
                        model_name=model.model_id,
                        timestamp=result.timestamp(),
                    )
                    yield _dump_message(response_message, message_index=next_message_index + 1)

                await _save_history(serialize_model_messages(result.all_messages()))
        except UnexpectedModelBehavior as e:
            log.warning(f'聊天模型响应异常，已降级为错误提示: conversation_id={conversation_id}, error={e}')
            failure_message = '服务暂时不可用，请稍后再试'
            failure_response = ModelResponse(
                parts=[TextPart(failure_message)],
                model_name=model.model_id,
                metadata={
                    'is_error': True,
                    'error_message': str(e),
                },
            )
            persisted_messages = [*message_history]
            if should_emit_user_message:
                persisted_messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
            persisted_messages.append(failure_response)
            yield _dump_message(failure_response, message_index=next_message_index + 1)
            await _save_history(serialize_model_messages(persisted_messages))


ai_chat_service: ChatService = ChatService()
