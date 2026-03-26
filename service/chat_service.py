import base64
import json

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from fastapi.sse import format_sse_event
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    BinaryContent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.core.conf import settings
from backend.plugin.ai.crud.crud_chat_history import ai_chat_history_dao
from backend.plugin.ai.crud.crud_chat_message import ai_chat_message_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import (
    AIChatAttachmentSourceType,
    AIChatAttachmentType,
    AIChatMessageRoleType,
    AIChatOutputModeType,
)
from backend.plugin.ai.schema.chat import AIChatParam, CreateAIChatParam, EditAIChatParam
from backend.plugin.ai.schema.chat_history import CreateAIChatHistoryParam, UpdateAIChatHistoryParam
from backend.plugin.ai.service.chat_history_service import ai_chat_history_service
from backend.plugin.ai.service.mcp_service import mcp_service
from backend.plugin.ai.tools.chat_builtin_tools import register_chat_builtin_tools
from backend.plugin.ai.utils.chat_control import build_model_settings, build_output_type
from backend.plugin.ai.utils.model_control import get_provider_model
from backend.utils.timezone import timezone


@dataclass(slots=True)
class ChatAgentDeps:
    """聊天代理依赖"""

    db: AsyncSession
    user_id: int


class ChatService:
    """聊天服务类"""

    @staticmethod
    async def _persist_chat_messages(
        *,
        db: AsyncSession,
        conversation_id: str,
        prompt: str,
        user_id: int,
        chat: AIChatParam,
        chat_history: Any,
        existing_message_rows: list[Any],
        preserved_prefix_count: int,
        persisted_messages: list[dict[str, object]],
    ) -> None:
        """
        持久化当前会话消息

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param prompt: 当前用户提示词
        :param user_id: 用户 ID
        :param chat: 聊天参数
        :param chat_history: 已存在的会话记录，不存在则创建新会话
        :param existing_message_rows: 数据库中的原始消息记录
        :param preserved_prefix_count: 需保留原始消息元信息的前缀长度
        :param persisted_messages: 最终需要落库的完整消息序列
        :return:
        """
        title = chat_history.title if chat_history else ' '.join(prompt.split())
        if not title:
            title = '新会话'
        elif len(title) > 256:
            title = title[:253] + '...'
        payload = {
            'conversation_id': conversation_id,
            'title': title,
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
        if persisted_messages:
            await ai_chat_message_dao.bulk_create(
                db,
                [
                    {
                        'conversation_id': conversation_id,
                        'provider_id': existing_message_rows[index].provider_id
                        if index < preserved_prefix_count and index < len(existing_message_rows)
                        else chat.provider_id,
                        'model_id': existing_message_rows[index].model_id
                        if index < preserved_prefix_count and index < len(existing_message_rows)
                        else chat.model_id,
                        'message_index': index,
                        'message': message,
                    }
                    for index, message in enumerate(persisted_messages)
                ],
            )

    async def stream_messages(  # noqa: C901
        self,
        *,
        db: AsyncSession,
        user_id: int,
        chat: AIChatParam,
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

        prepared = await ai_chat_history_service.prepare_chat_context(db=db, user_id=user_id, chat=chat)
        conversation_id = prepared.conversation_id
        chat_history = prepared.chat_history
        existing_message_rows = prepared.existing_message_rows
        message_history = prepared.message_history
        prompt = prepared.prompt
        next_message_index = prepared.next_message_index
        should_emit_user_message = prepared.should_emit_user_message
        preserved_prefix_count = prepared.preserved_prefix_count
        request_attachments = chat.attachments if isinstance(chat, (CreateAIChatParam, EditAIChatParam)) else None

        # 将前端附件参数归一化为 pydantic-ai 可直接消费的多模态输入对象
        attachments: list[Any] = []
        for attachment in request_attachments or []:
            if attachment.source_type == AIChatAttachmentSourceType.url:
                if not attachment.url:
                    raise errors.RequestError(msg='URL 类型附件必须提供 url')
                url = str(attachment.url)
                if attachment.type == AIChatAttachmentType.image:
                    attachments.append(ImageUrl(url))
                elif attachment.type == AIChatAttachmentType.audio:
                    attachments.append(AudioUrl(url))
                elif attachment.type == AIChatAttachmentType.video:
                    attachments.append(VideoUrl(url))
                else:
                    attachments.append(DocumentUrl(url))
                continue

            if not attachment.content:
                raise errors.RequestError(msg='Base64 类型附件必须提供 content')
            if not attachment.media_type:
                raise errors.RequestError(msg='Base64 类型附件必须提供 media_type')
            attachments.append(
                BinaryContent(
                    data=base64.b64decode(attachment.content),
                    media_type=attachment.media_type,
                )
            )

        user_input = [prompt, *attachments] if attachments else prompt
        model_settings = build_model_settings(chat=chat, provider_type=provider.type)
        output_type = build_output_type(chat=chat)
        toolsets = await mcp_service.get_toolsets(db=db, mcp_ids=chat.mcp_ids) if chat.mcp_ids else []
        model_instance = get_provider_model(
            provider_type=provider.type,
            model_name=model.model_id,
            api_key=provider.api_key,
            base_url=provider.api_host,
            model_settings=model_settings,
        )

        # 动态构建本次对话 Agent
        agent = Agent(
            name='fba_chat',
            deps_type=ChatAgentDeps,
            model=model_instance,
            output_type=output_type,
            toolsets=toolsets,
        )

        # 注册项目内置工具
        if chat.enable_builtin_tools:
            register_chat_builtin_tools(agent)

        # 对新提问/编辑重发场景，先补发一条用户消息事件，便于前端即时渲染会话列表
        if should_emit_user_message:
            yield format_sse_event(
                event='response.user',
                data_str=json.dumps(
                    {
                        'conversation_id': conversation_id,
                        'message_index': next_message_index,
                        'role': AIChatMessageRoleType.user,
                        'content': prompt,
                        'timestamp': timezone.now().isoformat(),
                    },
                    ensure_ascii=False,
                ),
            )

        for attempt in range(settings.AI_CHAT_MAX_RETRIES + 1):
            try:
                # 文本流模式下逐段转发模型输出，并把 thinking/content/tool 事件映射为前端 SSE 协议
                if chat.output_mode == AIChatOutputModeType.text:
                    part_message_indexes: dict[int, int] = {}
                    message_contents: dict[int, str] = {}
                    stream_message_index = next_message_index + (1 if should_emit_user_message else 0)
                    result = None

                    async for stream_event in agent.run_stream_events(
                        user_input,
                        message_history=message_history,
                        deps=ChatAgentDeps(db=db, user_id=user_id),
                    ):
                        if isinstance(stream_event, AgentRunResultEvent):
                            result = stream_event.result
                            continue

                        if isinstance(stream_event, (PartStartEvent, PartDeltaEvent, PartEndEvent)):
                            role: AIChatMessageRoleType | None = None
                            content = ''
                            event = ''

                            if isinstance(stream_event, PartStartEvent):
                                if isinstance(stream_event.part, ThinkingPart):
                                    if not chat.include_thinking:
                                        continue
                                    role = AIChatMessageRoleType.thinking
                                    content = stream_event.part.content
                                elif isinstance(stream_event.part, TextPart):
                                    role = AIChatMessageRoleType.model
                                    content = stream_event.part.content
                                event = 'created'
                            elif isinstance(stream_event, PartDeltaEvent):
                                if isinstance(stream_event.delta, ThinkingPartDelta):
                                    if not chat.include_thinking:
                                        continue
                                    role = AIChatMessageRoleType.thinking
                                    content = stream_event.delta.content_delta
                                elif isinstance(stream_event.delta, TextPartDelta):
                                    role = AIChatMessageRoleType.model
                                    content = stream_event.delta.content_delta
                                event = 'delta'
                            else:
                                if isinstance(stream_event.part, ThinkingPart):
                                    if not chat.include_thinking:
                                        continue
                                    role = AIChatMessageRoleType.thinking
                                    content = stream_event.part.content
                                elif isinstance(stream_event.part, TextPart):
                                    role = AIChatMessageRoleType.model
                                    content = stream_event.part.content
                                event = 'done'

                            if role is None or (event == 'delta' and not content):
                                continue

                            # 同一个 part 在 created/delta/done 生命周期内复用同一 message_index
                            message_index = part_message_indexes.get(stream_event.index)
                            if message_index is None:
                                message_index = stream_message_index
                                stream_message_index += 1
                                part_message_indexes[stream_event.index] = message_index

                            event_prefix = (
                                'response.reasoning'
                                if role == AIChatMessageRoleType.thinking
                                else 'response.output_text'
                            )
                            if event == 'created':
                                message_contents.setdefault(message_index, '')
                                yield format_sse_event(
                                    event=f'{event_prefix}.created',
                                    data_str=json.dumps(
                                        {
                                            'conversation_id': conversation_id,
                                            'message_index': message_index,
                                            'role': role,
                                            'timestamp': timezone.now().isoformat(),
                                        },
                                        ensure_ascii=False,
                                    ),
                                )
                                if content:
                                    message_contents[message_index] = content
                                    yield format_sse_event(
                                        event=f'{event_prefix}.delta',
                                        data_str=json.dumps(
                                            {
                                                'conversation_id': conversation_id,
                                                'message_index': message_index,
                                                'delta': content,
                                            },
                                            ensure_ascii=False,
                                        ),
                                    )
                            elif event == 'delta':
                                message_contents[message_index] = message_contents.get(message_index, '') + content
                                yield format_sse_event(
                                    event=f'{event_prefix}.delta',
                                    data_str=json.dumps(
                                        {
                                            'conversation_id': conversation_id,
                                            'message_index': message_index,
                                            'delta': content,
                                        },
                                        ensure_ascii=False,
                                    ),
                                )
                            else:
                                message_contents[message_index] = content
                                yield format_sse_event(
                                    event=f'{event_prefix}.done',
                                    data_str=json.dumps(
                                        {
                                            'conversation_id': conversation_id,
                                            'message_index': message_index,
                                            'content': content,
                                        },
                                        ensure_ascii=False,
                                    ),
                                )
                            continue

                        if isinstance(stream_event, FunctionToolCallEvent):
                            yield format_sse_event(
                                event='response.tool_call',
                                data_str=json.dumps(
                                    {
                                        'conversation_id': conversation_id,
                                        'tool_call_id': stream_event.part.tool_call_id,
                                        'tool_name': stream_event.part.tool_name,
                                        'args': stream_event.part.args,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            continue

                        if isinstance(stream_event, FunctionToolResultEvent):
                            yield format_sse_event(
                                event='response.tool_result',
                                data_str=json.dumps(
                                    {
                                        'conversation_id': conversation_id,
                                        'tool_call_id': stream_event.tool_call_id,
                                        'content': stream_event.result.content,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            continue

                        if isinstance(stream_event, FinalResultEvent):
                            yield format_sse_event(
                                event='response.final_result',
                                data_str=json.dumps(
                                    {
                                        'conversation_id': conversation_id,
                                        'tool_name': stream_event.tool_name,
                                        'tool_call_id': stream_event.tool_call_id,
                                    },
                                    ensure_ascii=False,
                                ),
                            )

                    if result is None:
                        raise errors.ServerError(msg='聊天流结束时未获取最终结果')

                    # 流式响应结束后，以模型最终完整消息序列为准落库，避免只保存增量片段。
                    persisted_messages = to_jsonable_python(list(result.all_messages()))
                    assert isinstance(persisted_messages, list)
                    await self._persist_chat_messages(
                        db=db,
                        conversation_id=conversation_id,
                        prompt=prompt,
                        user_id=user_id,
                        chat=chat,
                        chat_history=chat_history,
                        existing_message_rows=existing_message_rows,
                        preserved_prefix_count=preserved_prefix_count,
                        persisted_messages=persisted_messages,
                    )

                    yield format_sse_event(
                        event='response.completed',
                        data_str=json.dumps({'conversation_id': conversation_id}, ensure_ascii=False),
                    )
                else:
                    # 结构化输出模式不逐 token 推送，等待结果成型后一次性回传并持久化
                    async with agent.run_stream(
                        user_input, message_history=message_history, deps=ChatAgentDeps(db=db, user_id=user_id)
                    ) as result:
                        structured_output = await result.get_output()
                        serialized_output = to_jsonable_python(structured_output)
                        if not isinstance(serialized_output, (dict, list)):
                            serialized_output = {'value': serialized_output}
                        structured_response = ModelResponse(
                            parts=[TextPart(json.dumps(serialized_output, ensure_ascii=False))],
                            model_name=model.model_id,
                            metadata={'structured_data': serialized_output},
                        )
                        persisted_messages = [*message_history]
                        if should_emit_user_message:
                            persisted_messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
                        persisted_messages.append(structured_response)
                        persisted_messages = to_jsonable_python(list(persisted_messages))
                        assert isinstance(persisted_messages, list)
                        await self._persist_chat_messages(
                            db=db,
                            conversation_id=conversation_id,
                            prompt=prompt,
                            user_id=user_id,
                            chat=chat,
                            chat_history=chat_history,
                            existing_message_rows=existing_message_rows,
                            preserved_prefix_count=preserved_prefix_count,
                            persisted_messages=persisted_messages,
                        )

                        yield format_sse_event(
                            event='response.structured.done',
                            data_str=json.dumps(
                                {
                                    'conversation_id': conversation_id,
                                    'message_index': next_message_index + (1 if should_emit_user_message else 0),
                                    'content': structured_response.parts[0].content,
                                    'structured_data': structured_response.metadata['structured_data'],
                                },
                                ensure_ascii=False,
                            ),
                        )

                        yield format_sse_event(
                            event='response.completed',
                            data_str=json.dumps({'conversation_id': conversation_id}, ensure_ascii=False),
                        )

                break
            except UnexpectedModelBehavior as e:
                # 模型返回异常时先按配置重试；重试耗尽后降级为一条可展示、可追踪的错误消息。
                if attempt < settings.AI_CHAT_MAX_RETRIES:
                    log.warning(
                        f'聊天模型响应异常，准备重试: conversation_id={conversation_id}, '
                        f'attempt={attempt + 1}, error={e}'
                    )
                    yield format_sse_event(
                        event='response.retrying',
                        data_str=json.dumps(
                            {
                                'conversation_id': conversation_id,
                                'attempt': attempt + 1,
                                'max_retries': settings.AI_CHAT_MAX_RETRIES,
                                'message': '模型响应异常，正在重试',
                                'detail': str(e),
                            },
                            ensure_ascii=False,
                        ),
                    )
                    continue

                log.error(f'聊天模型响应异常，已降级为错误提示: conversation_id={conversation_id}, error={e}')
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
                persisted_messages = to_jsonable_python(list(persisted_messages))
                assert isinstance(persisted_messages, list)
                await self._persist_chat_messages(
                    db=db,
                    conversation_id=conversation_id,
                    prompt=prompt,
                    user_id=user_id,
                    chat=chat,
                    chat_history=chat_history,
                    existing_message_rows=existing_message_rows,
                    preserved_prefix_count=preserved_prefix_count,
                    persisted_messages=persisted_messages,
                )

                yield format_sse_event(
                    event='response.error',
                    data_str=json.dumps(
                        {
                            'conversation_id': conversation_id,
                            'message_index': next_message_index + (1 if should_emit_user_message else 0),
                            'message': failure_message,
                            'detail': str(e),
                        },
                        ensure_ascii=False,
                    ),
                )

                yield format_sse_event(
                    event='response.completed',
                    data_str=json.dumps({'conversation_id': conversation_id}, ensure_ascii=False),
                )

                break


ai_chat_service: ChatService = ChatService()
