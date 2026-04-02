from base64 import b64decode
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from typing import Any

from ag_ui.core import BinaryInputContent, RunAgentInput, RunErrorEvent, TextInputContent, UserMessage
from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import async_db_session, uuid4_str
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIChatGenerationType, AIProviderType
from backend.plugin.ai.schema.chat import AIChatCompletionParam, AIChatForwardedPropsParam, AIChatRegenerateParam
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.service.mcp_service import mcp_service
from backend.plugin.ai.tools.chat_builtin_tools import register_chat_builtin_tools
from backend.plugin.ai.utils.chat_control import build_model_settings
from backend.plugin.ai.utils.model_control import get_provider_model
from backend.plugin.ai.utils.web_search import build_chat_search_tools


@dataclass(slots=True)
class ChatAgentDeps:
    """聊天代理依赖"""

    db: AsyncSession
    user_id: int


@dataclass(slots=True)
class ChatConversationState:
    """聊天上下文状态"""

    conversation: Any | None
    message_rows: list[Any]
    model_messages: list[Any]
    context_start_index: int


@dataclass(slots=True)
class ChatCompletionPersistence:
    """聊天结果持久化上下文"""

    conversation_id: str
    user_id: int
    forwarded_props: AIChatForwardedPropsParam
    conversation: Any | None
    title: str
    replace_message_row_ids: list[int] | None
    replace_start_message_index: int | None
    replace_end_message_index: int | None
    base_message_index: int
    result_offset: int


class ChatService:
    """聊天服务"""

    FILE_URL_CONSTRUCTORS = {
        'image': ImageUrl,
        'video': VideoUrl,
        'audio': AudioUrl,
    }

    @staticmethod
    def _normalize_title(title: str) -> str:
        """
        规范化对话标题

        :param title: 原始标题
        :return:
        """
        normalized_title = title or '新对话'
        if len(normalized_title) > 256:
            normalized_title = normalized_title[:253] + '...'
        return normalized_title

    @staticmethod
    def is_user_prompt_message(message: Any) -> bool:
        """
        判断是否为用户输入消息

        :param message: 模型消息
        :return:
        """
        return (
            isinstance(message, ModelRequest) and bool(message.parts) and isinstance(message.parts[0], UserPromptPart)
        )

    @staticmethod
    def _build_binary_user_content(
        part: BinaryInputContent,
    ) -> BinaryContent | ImageUrl | AudioUrl | VideoUrl | DocumentUrl:
        """
        构建二进制用户输入内容

        :param part: AG-UI 二进制输入
        :return:
        """
        vendor_metadata = {'filename': part.filename} if part.filename else None
        if part.url:
            try:
                parsed_binary = BinaryContent.from_data_uri(part.url)
            except ValueError:
                media_type_prefix = part.mime_type.split('/', 1)[0]
                constructor = ChatService.FILE_URL_CONSTRUCTORS.get(media_type_prefix, DocumentUrl)
                return constructor(
                    url=part.url,
                    media_type=part.mime_type,
                    identifier=part.id,
                    vendor_metadata=vendor_metadata,
                )
            return BinaryContent.narrow_type(
                BinaryContent(
                    data=parsed_binary.data,
                    media_type=parsed_binary.media_type,
                    identifier=part.id,
                    vendor_metadata=vendor_metadata,
                )
            )
        if part.data:
            return BinaryContent.narrow_type(
                BinaryContent(
                    data=b64decode(part.data),
                    media_type=part.mime_type,
                    identifier=part.id,
                    vendor_metadata=vendor_metadata,
                )
            )
        raise errors.RequestError(msg='聊天消息格式非法')

    @staticmethod
    def _extract_prompt(first_part: UserPromptPart) -> tuple[str, bool]:
        """
        提取用户输入中的文本提示和二进制输入标记

        :param first_part: 用户输入内容块
        :return:
        """
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
        return prompt, has_binary_input

    @staticmethod
    def _load_current_user_message(message: UserMessage) -> ModelRequest:
        """
        解析当前轮用户消息，保留文件标识和文件名

        :param message: 用户消息
        :return:
        """
        content = message.content
        if isinstance(content, str):
            return ModelRequest(parts=[UserPromptPart(content=content)])

        user_prompt_content: list[Any] = []
        for part in content:
            if isinstance(part, TextInputContent):
                user_prompt_content.append(part.text)
                continue
            if not isinstance(part, BinaryInputContent):
                raise errors.RequestError(msg='聊天消息格式非法')
            user_prompt_content.append(ChatService._build_binary_user_content(part))

        if not user_prompt_content:
            raise errors.RequestError(msg='聊天消息不能为空')

        prompt_content = (
            user_prompt_content[0]
            if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str)
            else user_prompt_content
        )
        return ModelRequest(parts=[UserPromptPart(content=prompt_content)])

    @staticmethod
    def _prepare_run_input(
        *,
        thread_id: str | None,
        forwarded_props: AIChatForwardedPropsParam,
        default_conversation_id: str | None = None,
        expected_conversation_id: str | None = None,
    ) -> RunAgentInput:
        """
        解析并补全运行参数

        :param thread_id: 对话 ID
        :param forwarded_props: 聊天扩展参数
        :param default_conversation_id: 默认对话 ID
        :param expected_conversation_id: 期望对话 ID
        :return:
        """
        conversation_id = thread_id or default_conversation_id or uuid4_str()
        run_input = RunAgentInput.model_validate({
            'thread_id': conversation_id,
            'run_id': uuid4_str(),
            'parent_run_id': None,
            'state': {},
            'messages': [],
            'tools': [],
            'context': [],
            'forwarded_props': forwarded_props.model_dump(),
        })

        if expected_conversation_id is not None and run_input.thread_id != expected_conversation_id:
            raise errors.RequestError(msg='请求体中的对话 ID 与路径不一致')

        return run_input

    @staticmethod
    async def _build_agent(*, db: AsyncSession, forwarded_props: AIChatForwardedPropsParam) -> Agent:
        """
        构建聊天代理

        :param db: 数据库会话
        :param forwarded_props: 聊天扩展参数
        :return:
        """

        provider = await ai_provider_dao.get(db, forwarded_props.provider_id)
        if not provider:
            raise errors.NotFoundError(msg='供应商不存在')
        if not provider.status:
            raise errors.RequestError(msg='此供应商暂不可用，请更换供应商或联系系统管理员')
        if forwarded_props.generation_type == AIChatGenerationType.image and provider.type != AIProviderType.google:
            raise errors.RequestError(msg='当前图片生成仅支持 Google 供应商')

        model = await ai_model_dao.get_by_model_and_provider(db, forwarded_props.model_id, forwarded_props.provider_id)
        if not model:
            raise errors.NotFoundError(msg='供应商模型不存在')
        if not model.status:
            raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')

        model_settings = build_model_settings(chat_metadata=forwarded_props, provider_type=provider.type)
        toolsets = (
            await mcp_service.get_toolsets(db=db, mcp_ids=forwarded_props.mcp_ids) if forwarded_props.mcp_ids else None
        )
        tools, builtin_tools = build_chat_search_tools(
            web_search=forwarded_props.web_search,
            provider_type=provider.type,
        )
        if forwarded_props.generation_type == AIChatGenerationType.image:
            builtin_tools = [*builtin_tools, ImageGenerationTool()]
        model_instance = get_provider_model(
            provider_type=provider.type,
            model_name=model.model_id,
            api_key=provider.api_key,
            base_url=provider.api_host,
            model_settings=model_settings,
        )

        agent = Agent(
            name='fba_chat',
            deps_type=ChatAgentDeps,
            model=model_instance,
            output_type=[BinaryImage, str] if forwarded_props.generation_type == AIChatGenerationType.image else str,
            tools=tools,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )
        if forwarded_props.enable_builtin_tools:
            register_chat_builtin_tools(agent)
        return agent

    @staticmethod
    async def _load_conversation_state(
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        must_exist: bool,
        require_messages: bool = False,
    ) -> ChatConversationState:
        """
        加载对话状态

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param must_exist: 对话是否必须存在
        :param require_messages: 是否要求对话消息存在
        :return:
        """

        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation:
            if must_exist:
                raise errors.NotFoundError(msg='对话不存在')
            return ChatConversationState(
                conversation=None,
                message_rows=[],
                model_messages=[],
                context_start_index=0,
            )
        if conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')

        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        if require_messages and not message_rows:
            raise errors.RequestError(msg='对话消息不存在')

        context_start_index = 0
        if conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1

        model_messages = (
            list(
                ModelMessagesTypeAdapter.validate_python(
                    [row.message for row in message_rows],
                )
            )
            if message_rows
            else []
        )

        return ChatConversationState(
            conversation=conversation,
            message_rows=message_rows,
            model_messages=model_messages,
            context_start_index=context_start_index,
        )

    async def _persist_completion_messages(
        self,
        *,
        db: AsyncSession,
        persistence: ChatCompletionPersistence,
        messages: list[Any],
    ) -> None:
        """
        持久化模型输出消息

        :param db: 数据库会话
        :param persistence: 持久化上下文
        :param messages: 待持久化消息
        :return:
        """
        if not messages:
            return

        payload_messages = to_jsonable_python(messages)
        assert isinstance(payload_messages, list)

        insert_message_index = persistence.base_message_index
        current_conversation = persistence.conversation or await ai_conversation_dao.get_by_conversation_id(
            db, persistence.conversation_id
        )
        normalized_title = self._normalize_title(persistence.title)

        payload = {
            'conversation_id': persistence.conversation_id,
            'title': normalized_title,
            'provider_id': persistence.forwarded_props.provider_id,
            'model_id': persistence.forwarded_props.model_id,
            'user_id': current_conversation.user_id if current_conversation else persistence.user_id,
            'pinned_time': current_conversation.pinned_time if current_conversation else None,
            'context_start_message_id': current_conversation.context_start_message_id if current_conversation else None,
            'context_cleared_time': current_conversation.context_cleared_time if current_conversation else None,
        }
        if current_conversation:
            await ai_conversation_dao.update(db, current_conversation.id, UpdateAIConversationParam(**payload))
        else:
            await ai_conversation_dao.create(db, CreateAIConversationParam(**payload))

        if (
            persistence.replace_message_row_ids is not None
            and persistence.replace_start_message_index is not None
            and persistence.replace_end_message_index is not None
        ):
            replace_count = persistence.replace_end_message_index - persistence.replace_start_message_index + 1
            shared_count = min(replace_count, len(payload_messages))

            for index in range(shared_count):
                await ai_message_dao.update(
                    db,
                    persistence.replace_message_row_ids[index],
                    {
                        'provider_id': persistence.forwarded_props.provider_id,
                        'model_id': persistence.forwarded_props.model_id,
                        'message_index': persistence.replace_start_message_index + index,
                        'message': payload_messages[index],
                    },
                )

            if len(payload_messages) < replace_count:
                await ai_message_dao.delete_message_index_range(
                    db,
                    persistence.conversation_id,
                    persistence.replace_start_message_index + len(payload_messages),
                    persistence.replace_end_message_index,
                )
                await ai_message_dao.update_message_indexes_offset(
                    db,
                    persistence.conversation_id,
                    persistence.replace_end_message_index + 1,
                    len(payload_messages) - replace_count,
                )
                return

            if len(payload_messages) == replace_count:
                return

            await ai_message_dao.update_message_indexes_offset(
                db,
                persistence.conversation_id,
                persistence.replace_end_message_index + 1,
                len(payload_messages) - replace_count,
            )
            payload_messages = payload_messages[replace_count:]
            insert_message_index = persistence.replace_end_message_index + 1

        await ai_message_dao.bulk_create(
            db,
            [
                {
                    'conversation_id': persistence.conversation_id,
                    'provider_id': persistence.forwarded_props.provider_id,
                    'model_id': persistence.forwarded_props.model_id,
                    'message_index': insert_message_index + index,
                    'message': message,
                }
                for index, message in enumerate(payload_messages)
            ],
        )

    async def _persist_completion_result(
        self,
        result: Any,
        *,
        db: AsyncSession,
        persistence: ChatCompletionPersistence,
    ) -> None:
        """
        持久化成功完成的聊天结果

        :param result: 运行结果
        :param db: 数据库会话
        :param persistence: 持久化上下文
        :return:
        """
        persisted_messages = to_jsonable_python(list(result.all_messages()))
        assert isinstance(persisted_messages, list)
        await self._persist_completion_messages(
            db=db,
            persistence=persistence,
            messages=persisted_messages[persistence.result_offset :],
        )

    def _stream_response(
        self,
        *,
        db: AsyncSession,
        user_id: int,
        agent: Agent,
        run_input: RunAgentInput,
        accept: str | None,
        message_history: list[Any],
        on_complete: Any,
        persistence: ChatCompletionPersistence,
    ) -> StreamingResponse:
        """
        运行聊天代理并返回流式响应

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param agent: 聊天代理
        :param run_input: 运行参数
        :param accept: Accept 请求头
        :param message_history: 消息历史
        :param on_complete: 完成回调
        :param persistence: 持久化上下文
        :return:
        """

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(
            deps=ChatAgentDeps(db=db, user_id=user_id),
            message_history=message_history,
            on_complete=on_complete,
        )

        async def stream_with_error_persistence() -> AsyncIterator[Any]:
            error_persisted = False
            async for event in event_stream:
                if isinstance(event, RunErrorEvent) and not error_persisted:
                    error_persisted = True
                    raw_error_message = event.message.strip() if event.message else ''
                    try:
                        error_message = raw_error_message or '模型请求失败，请稍后重试'
                        await self._persist_completion_messages(
                            db=db,
                            persistence=persistence,
                            messages=[
                                ModelResponse(
                                    parts=[TextPart(content=f'模型请求失败：{error_message}')],
                                    model_name=persistence.forwarded_props.model_id,
                                    metadata={
                                        'is_error': True,
                                        'error_message': error_message,
                                    },
                                )
                            ],
                        )
                    except Exception as e:
                        log.exception(f'持久化聊天失败消息异常: {e}')
                    else:
                        log.warning(
                            f'聊天运行失败，已写入对话记录 conversation_id={persistence.conversation_id}: {raw_error_message}'  # noqa: E501
                        )
                yield event

        response = adapter.streaming_response(stream_with_error_persistence())
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Cache-Control'] = 'no-cache'
        return response

    async def create_completion(
        self, *, db: AsyncSession, user_id: int, obj: AIChatCompletionParam, accept: str | None
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
            current_message = self._load_current_user_message(obj.message)
        except Exception as e:
            log.warning(f'聊天消息加载失败: {e}')
            if isinstance(e, errors.BaseExceptionError):
                raise
            raise errors.RequestError(msg='聊天消息格式非法') from e
        if not current_message.parts:
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')
        first_part = current_message.parts[0]
        if not isinstance(first_part, UserPromptPart):
            raise errors.RequestError(msg='普通聊天请求仅支持传入当前轮用户消息')

        prompt, has_binary_input = self._extract_prompt(first_part)
        if not prompt and not has_binary_input:
            raise errors.RequestError(msg='当前轮用户消息不能为空')

        run_input = self._prepare_run_input(
            thread_id=obj.thread_id,
            forwarded_props=obj.forwarded_props,
        )

        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        agent = await self._build_agent(db=db, forwarded_props=forwarded_props)

        conversation_id = run_input.thread_id
        payload_messages = to_jsonable_python([current_message])
        assert isinstance(payload_messages, list)

        # 使用独立事务先提交用户输入，避免流式阶段异常导致整段会话回滚
        async with async_db_session.begin() as session:
            conversation = await ai_conversation_dao.get_by_conversation_id(session, conversation_id)
            if conversation and conversation.user_id != user_id:
                raise errors.NotFoundError(msg='对话不存在')

            payload = {
                'conversation_id': conversation_id,
                'title': conversation.title if conversation else self._normalize_title(prompt),
                'provider_id': forwarded_props.provider_id,
                'model_id': forwarded_props.model_id,
                'user_id': conversation.user_id if conversation else user_id,
                'pinned_time': conversation.pinned_time if conversation else None,
                'context_start_message_id': conversation.context_start_message_id if conversation else None,
                'context_cleared_time': conversation.context_cleared_time if conversation else None,
            }
            if conversation:
                await ai_conversation_dao.update(session, conversation.id, UpdateAIConversationParam(**payload))
            else:
                await ai_conversation_dao.create(session, CreateAIConversationParam(**payload))

            message_rows = list(await ai_message_dao.get_all(session, conversation_id))
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
        state = await self._load_conversation_state(
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
            base_message_index=len(state.message_rows),
            result_offset=len(message_history),
        )

        return self._stream_response(
            db=db,
            user_id=user_id,
            agent=agent,
            run_input=run_input,
            accept=accept,
            message_history=message_history,
            on_complete=partial(self._persist_completion_result, db=db, persistence=persistence),
            persistence=persistence,
        )

    @staticmethod
    def _get_message_index(*, message_rows: list[Any], message_id: int) -> int:
        """
        查找消息索引

        :param message_rows: 消息记录列表
        :param message_id: 消息 ID
        :return:
        """

        target_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if target_index is None:
            raise errors.NotFoundError(msg='消息不存在')
        return target_index

    def _get_reply_end_index(self, *, model_messages: list[Any], reply_start_index: int) -> int:
        """
        获取当前回复轮次结束索引

        :param model_messages: 模型消息列表
        :param reply_start_index: 回复起始索引
        :return:
        """
        next_user_message_index = next(
            (
                index
                for index in range(reply_start_index + 1, len(model_messages))
                if self.is_user_prompt_message(model_messages[index])
            ),
            None,
        )
        return (next_user_message_index - 1) if next_user_message_index is not None else len(model_messages) - 1

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
        run_input = self._prepare_run_input(
            thread_id=obj.thread_id,
            forwarded_props=obj.forwarded_props,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )

        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        agent = await self._build_agent(db=db, forwarded_props=forwarded_props)
        state = await self._load_conversation_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )

        target_index = self._get_message_index(message_rows=state.message_rows, message_id=message_id)
        target_message = state.model_messages[target_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if not self.is_user_prompt_message(target_message):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')
        reply_start_index = target_index + 1
        if reply_start_index >= len(state.model_messages):
            raise errors.RequestError(msg='当前消息后不存在可重生成的 AI 回复')
        reply_end_index = self._get_reply_end_index(
            model_messages=state.model_messages,
            reply_start_index=reply_start_index,
        )

        message_history = state.model_messages[state.context_start_index : target_index + 1]
        persistence = ChatCompletionPersistence(
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title,
            replace_message_row_ids=[row.id for row in state.message_rows[reply_start_index : reply_end_index + 1]],
            replace_start_message_index=reply_start_index,
            replace_end_message_index=reply_end_index,
            base_message_index=reply_start_index,
            result_offset=len(message_history),
        )

        return self._stream_response(
            db=db,
            user_id=user_id,
            agent=agent,
            run_input=run_input,
            accept=accept,
            message_history=message_history,
            on_complete=partial(self._persist_completion_result, db=db, persistence=persistence),
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
        run_input = self._prepare_run_input(
            thread_id=obj.thread_id,
            forwarded_props=obj.forwarded_props,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )

        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        agent = await self._build_agent(db=db, forwarded_props=forwarded_props)
        state = await self._load_conversation_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )

        target_index = self._get_message_index(message_rows=state.message_rows, message_id=message_id)
        if not isinstance(state.model_messages[target_index], ModelResponse):
            raise errors.RequestError(msg='仅支持根据 AI 回复重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        user_message_index = next(
            (
                index
                for index in range(target_index - 1, state.context_start_index - 1, -1)
                if self.is_user_prompt_message(state.model_messages[index])
            ),
            None,
        )
        if user_message_index is None:
            raise errors.RequestError(msg='未找到对应的用户消息')
        reply_start_index = user_message_index + 1
        reply_end_index = self._get_reply_end_index(
            model_messages=state.model_messages,
            reply_start_index=reply_start_index,
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
            base_message_index=reply_start_index,
            result_offset=len(message_history),
        )

        return self._stream_response(
            db=db,
            user_id=user_id,
            agent=agent,
            run_input=run_input,
            accept=accept,
            message_history=message_history,
            on_complete=partial(self._persist_completion_result, db=db, persistence=persistence),
            persistence=persistence,
        )


ai_chat_service: ChatService = ChatService()
