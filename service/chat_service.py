from dataclasses import dataclass
from typing import Any

from fastapi.responses import Response
from pydantic_ai import Agent, BinaryImage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.database.db import uuid4_str
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIChatGenerationType, AIChatOutputModeType, AIProviderType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam
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


class ChatService:
    """聊天服务"""

    @staticmethod
    def _prepare_run_input(
        body: bytes,
        *,
        default_conversation_id: str | None = None,
        expected_conversation_id: str | None = None,
    ) -> Any:
        """
        解析并补全 AG-UI 运行参数

        :param body: 请求体
        :param default_conversation_id: 默认对话 ID
        :param expected_conversation_id: 期望对话 ID
        :return:
        """
        run_input = AGUIAdapter.build_run_input(body)

        update: dict[str, Any] = {}
        if not run_input.thread_id:
            update['thread_id'] = default_conversation_id or uuid4_str()
        if not run_input.run_id:
            update['run_id'] = uuid4_str()
        if update:
            run_input = run_input.model_copy(update=update)

        if expected_conversation_id is not None and run_input.thread_id != expected_conversation_id:
            raise errors.RequestError(msg='请求体中的对话 ID 与路径不一致')

        return run_input

    @staticmethod
    def _parse_forwarded_props(run_input: Any) -> AIChatForwardedPropsParam:
        """
        解析聊天扩展参数

        :param run_input: AG-UI 运行参数
        :return:
        """
        forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        if forwarded_props.output_mode != AIChatOutputModeType.text:
            raise errors.RequestError(msg='当前聊天仅支持文本响应')
        if (
            forwarded_props.output_schema
            or forwarded_props.output_schema_name
            or forwarded_props.output_schema_description
        ):
            raise errors.RequestError(msg='当前聊天接口暂不支持结构化输出')
        return forwarded_props

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
            raise errors.RequestError(msg='当前仅支持 Google 图片生成模型')

        model = await ai_model_dao.get_by_model_and_provider(db, forwarded_props.model_id, forwarded_props.provider_id)
        if not model:
            raise errors.NotFoundError(msg='供应商模型不存在')
        if not model.status:
            raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')

        model_settings = build_model_settings(chat=forwarded_props, provider_type=provider.type)
        toolsets = (
            await mcp_service.get_toolsets(db=db, mcp_ids=forwarded_props.mcp_ids) if forwarded_props.mcp_ids else []
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
        加载对话与消息上下文

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

        model_messages = (
            list(
                ModelMessagesTypeAdapter.validate_python(
                    [row.message for row in message_rows],
                )
            )
            if message_rows
            else []
        )
        context_start_index = 0
        if conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1

        return ChatConversationState(
            conversation=conversation,
            message_rows=message_rows,
            model_messages=model_messages,
            context_start_index=context_start_index,
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

    @staticmethod
    def _build_completion_callback(
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        forwarded_props: AIChatForwardedPropsParam,
        conversation: Any | None,
        title: str,
        delete_from_message_index: int | None,
        base_message_index: int,
        result_offset: int,
    ) -> Any:
        """
        构建完成后持久化回调

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param forwarded_props: 聊天扩展参数
        :param conversation: 对话对象
        :param title: 对话标题
        :param delete_from_message_index: 删除消息起始索引
        :param base_message_index: 新消息起始索引
        :param result_offset: 结果偏移量
        :return:
        """

        async def on_complete(result: Any) -> None:
            persisted_messages = to_jsonable_python(list(result.all_messages()))
            assert isinstance(persisted_messages, list)

            normalized_title = title or '新对话'
            if len(normalized_title) > 256:
                normalized_title = normalized_title[:253] + '...'

            payload = {
                'conversation_id': conversation_id,
                'title': normalized_title,
                'provider_id': forwarded_props.provider_id,
                'model_id': forwarded_props.model_id,
                'user_id': conversation.user_id if conversation else user_id,
                'pinned_time': conversation.pinned_time if conversation else None,
                'context_start_message_id': conversation.context_start_message_id if conversation else None,
                'context_cleared_time': conversation.context_cleared_time if conversation else None,
            }
            if conversation:
                await ai_conversation_dao.update(db, conversation.id, UpdateAIConversationParam(**payload))
            else:
                await ai_conversation_dao.create(db, CreateAIConversationParam(**payload))

            if delete_from_message_index is not None:
                await ai_message_dao.delete_after_message_index(db, conversation_id, delete_from_message_index)

            tail_messages = persisted_messages[result_offset:]
            if not tail_messages:
                return

            await ai_message_dao.bulk_create(
                db,
                [
                    {
                        'conversation_id': conversation_id,
                        'provider_id': forwarded_props.provider_id,
                        'model_id': forwarded_props.model_id,
                        'message_index': base_message_index + index,
                        'message': message,
                    }
                    for index, message in enumerate(tail_messages)
                ],
            )

        return on_complete

    @staticmethod
    def _stream_response(
        *,
        agent: Agent,
        run_input: Any,
        accept: str | None,
        db: AsyncSession,
        user_id: int,
        message_history: list[Any],
        on_complete: Any,
    ) -> Response:
        """
        运行聊天代理并返回流式响应

        :param agent: 聊天代理
        :param run_input: AG-UI 运行参数
        :param accept: Accept 请求头
        :param db: 数据库会话
        :param user_id: 用户 ID
        :param message_history: 消息历史
        :param on_complete: 完成回调
        :return:
        """

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(
            deps=ChatAgentDeps(db=db, user_id=user_id),
            message_history=message_history,
            on_complete=on_complete,
        )
        return adapter.streaming_response(event_stream)

    @staticmethod
    async def create_completion(*, db: AsyncSession, user_id: int, body: bytes, accept: str | None) -> Response:
        """
        创建流式对话

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param body: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_input = ChatService._prepare_run_input(body)

        forwarded_props = ChatService._parse_forwarded_props(run_input)
        if not run_input.messages:
            raise errors.RequestError(msg='聊天消息不能为空')

        try:
            input_messages = list(AGUIAdapter.load_messages(run_input.messages))
        except Exception as e:
            log.warning(f'AG-UI messages parse failed: error={e}')
            raise errors.RequestError(msg='AG-UI 消息格式非法') from e
        if not input_messages:
            raise errors.RequestError(msg='聊天消息不能为空')

        last_message = input_messages[-1]
        if not isinstance(last_message, ModelRequest) or not last_message.parts:
            raise errors.RequestError(msg='最后一条消息必须是用户消息')
        first_part = last_message.parts[0]
        if not isinstance(first_part, UserPromptPart):
            raise errors.RequestError(msg='最后一条消息必须是用户消息')

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
            raise errors.RequestError(msg='最后一条用户消息不能为空')

        conversation_id = run_input.thread_id
        agent = await ChatService._build_agent(db=db, forwarded_props=forwarded_props)

        state = await ChatService._load_conversation_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=False,
        )
        context_messages = state.model_messages[state.context_start_index :]
        message_history = [*context_messages, last_message] if state.conversation else input_messages

        on_complete = ChatService._build_completion_callback(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title if state.conversation else prompt,
            delete_from_message_index=len(state.message_rows) if state.conversation else None,
            base_message_index=len(state.message_rows),
            result_offset=len(context_messages) if state.conversation else 0,
        )
        return ChatService._stream_response(
            agent=agent,
            run_input=run_input,
            accept=accept,
            db=db,
            user_id=user_id,
            message_history=message_history,
            on_complete=on_complete,
        )

    @staticmethod
    async def regenerate_from_user_message(
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        message_id: int,
        body: bytes,
        accept: str | None,
    ) -> Response:
        """
        根据用户消息重生成 AI 回复

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param message_id: 消息 ID
        :param body: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_input = ChatService._prepare_run_input(
            body,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )

        forwarded_props = ChatService._parse_forwarded_props(run_input)
        agent = await ChatService._build_agent(db=db, forwarded_props=forwarded_props)
        state = await ChatService._load_conversation_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )

        target_index = ChatService._get_message_index(message_rows=state.message_rows, message_id=message_id)
        target_message = state.model_messages[target_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        message_history = state.model_messages[state.context_start_index : target_index + 1]
        preserved_history_count = target_index + 1
        on_complete = ChatService._build_completion_callback(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title,
            delete_from_message_index=preserved_history_count,
            base_message_index=preserved_history_count,
            result_offset=len(message_history),
        )
        return ChatService._stream_response(
            agent=agent,
            run_input=run_input,
            accept=accept,
            db=db,
            user_id=user_id,
            message_history=message_history,
            on_complete=on_complete,
        )

    @staticmethod
    async def regenerate_from_response_message(
        *,
        db: AsyncSession,
        user_id: int,
        conversation_id: str,
        message_id: int,
        body: bytes,
        accept: str | None,
    ) -> Response:
        """
        根据 AI 回复重生成

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param conversation_id: 对话 ID
        :param message_id: 消息 ID
        :param body: 请求体
        :param accept: Accept 请求头
        :return:
        """
        run_input = ChatService._prepare_run_input(
            body,
            default_conversation_id=conversation_id,
            expected_conversation_id=conversation_id,
        )

        forwarded_props = ChatService._parse_forwarded_props(run_input)
        agent = await ChatService._build_agent(db=db, forwarded_props=forwarded_props)
        state = await ChatService._load_conversation_state(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=True,
            require_messages=True,
        )

        target_index = ChatService._get_message_index(message_rows=state.message_rows, message_id=message_id)
        if isinstance(state.model_messages[target_index], ModelRequest):
            raise errors.RequestError(msg='仅支持根据 AI 回复重生成')
        if target_index < state.context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        user_message_index = next(
            (
                index
                for index in range(target_index - 1, state.context_start_index - 1, -1)
                if isinstance(state.model_messages[index], ModelRequest)
            ),
            None,
        )
        if user_message_index is None:
            raise errors.RequestError(msg='未找到对应的用户消息')

        message_history = state.model_messages[state.context_start_index : user_message_index + 1]
        preserved_history_count = user_message_index + 1
        on_complete = ChatService._build_completion_callback(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            forwarded_props=forwarded_props,
            conversation=state.conversation,
            title=state.conversation.title,
            delete_from_message_index=preserved_history_count,
            base_message_index=preserved_history_count,
            result_offset=len(message_history),
        )
        return ChatService._stream_response(
            agent=agent,
            run_input=run_input,
            accept=accept,
            db=db,
            user_id=user_id,
            message_history=message_history,
            on_complete=on_complete,
        )


ai_chat_service: ChatService = ChatService()
