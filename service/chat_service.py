import json

from collections.abc import AsyncGenerator
from typing import Any

from pydantic_ai import Agent, ModelResponse, ModelSettings, TextPart
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.schema.chat import AIChat
from backend.plugin.ai.utils.message_parse import to_chat_message
from backend.plugin.ai.utils.model_control import get_pydantic_model

chat_agent = Agent(name='fba_chat')


class ChatService:
    """聊天服务类"""

    @staticmethod
    async def stream_messages(*, db: AsyncSession, chat: AIChat) -> AsyncGenerator[bytes, Any]:
        """
        流式消息

        :param db: 数据库会话
        :param chat: 聊天参数
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

        yield json.dumps({'role': 'user', 'content': chat.user_prompt}, ensure_ascii=False).encode('utf-8') + b'\n'

        model_settings = {
            k: v
            for k, v in {
                'max_tokens': chat.max_tokens,
                'temperature': chat.temperature,
                'top_p': chat.top_p,
                'timeout': chat.timeout,
                'parallel_tool_calls': chat.parallel_tool_calls,
                'seed': chat.seed,
                'presence_penalty': chat.presence_penalty,
                'frequency_penalty': chat.frequency_penalty,
                'logit_bias': chat.logit_bias,
                'stop_sequences': chat.stop_sequences,
                'extra_headers': chat.extra_headers,
                'extra_body': chat.extra_body,
            }.items()
            if v is not None
        }

        async with chat_agent.run_stream(
            chat.user_prompt,
            model=get_pydantic_model(
                provider_type=provider.type,
                model_name=model.model_id,
                api_key=provider.api_key,
                base_url=provider.api_host,
                model_settings=ModelSettings(**model_settings),
            ),
        ) as result:
            async for text in result.stream_output(debounce_by=0.01):
                message = ModelResponse(parts=[TextPart(text)], model_name=model.model_id, timestamp=result.timestamp())
                yield json.dumps(to_chat_message(message)).encode('utf-8') + b'\n'


ai_chat_service: ChatService = ChatService()
