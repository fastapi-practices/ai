from dataclasses import dataclass
from typing import Any

from fastapi.responses import Response
from pydantic import ValidationError
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


class ChatService:
    """聊天服务"""

    async def create_completion(  # noqa: C901
        self,
        *,
        db: AsyncSession,
        user_id: int,
        body: bytes,
        accept: str | None,
    ) -> Response:
        """
        创建流式对话

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param body: 请求体
        :param accept: Accept 请求头
        :return:
        """
        try:
            run_input = AGUIAdapter.build_run_input(body)
        except ValidationError as e:
            return Response(content=e.json(), media_type='application/json', status_code=422)

        updates: dict[str, Any] = {}
        if not run_input.thread_id:
            updates['thread_id'] = uuid4_str()
        if not run_input.run_id:
            updates['run_id'] = uuid4_str()
        if updates:
            run_input = run_input.model_copy(update=updates)

        try:
            forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        except ValidationError as e:
            raise errors.RequestError(msg=f'聊天扩展参数非法: {e.errors()[0]["msg"]}') from e
        if forwarded_props.mode != 'create':
            raise errors.RequestError(msg='当前聊天接口仅支持 create 模式')
        if forwarded_props.output_mode != AIChatOutputModeType.text:
            raise errors.RequestError(msg='当前聊天接口仅支持文本输出模式')
        if (
            forwarded_props.output_schema
            or forwarded_props.output_schema_name
            or forwarded_props.output_schema_description
        ):
            raise errors.RequestError(msg='当前聊天接口暂不支持结构化输出')
        if not run_input.messages:
            raise errors.RequestError(msg='聊天消息不能为空')

        try:
            input_messages = AGUIAdapter.load_messages(run_input.messages)
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

        conversation = await ai_conversation_dao.get_by_conversation_id(db, run_input.thread_id)
        if conversation and conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        message_rows = list(await ai_message_dao.get_all(db, run_input.thread_id)) if conversation else []
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        context_start_index = 0
        if conversation and conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1

        message_history = [*model_messages[context_start_index:], last_message] if conversation else input_messages
        preserved_history_count = len(message_rows)
        context_message_count = len(model_messages[context_start_index:]) if conversation else 0

        async def on_complete(result: Any) -> None:
            persisted_messages = to_jsonable_python(list(result.all_messages()))
            assert isinstance(persisted_messages, list)

            title = conversation.title if conversation else prompt
            if not title:
                title = '新对话'
            elif len(title) > 256:
                title = title[:253] + '...'

            payload = {
                'conversation_id': run_input.thread_id,
                'title': title,
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

            if conversation and preserved_history_count > 0:
                await ai_message_dao.delete_after_message_index(db, run_input.thread_id, preserved_history_count)
                tail_messages = persisted_messages[context_message_count:]
            else:
                tail_messages = persisted_messages
            if tail_messages:
                await ai_message_dao.bulk_create(
                    db,
                    [
                        {
                            'conversation_id': run_input.thread_id,
                            'provider_id': forwarded_props.provider_id,
                            'model_id': forwarded_props.model_id,
                            'message_index': preserved_history_count + index,
                            'message': message,
                        }
                        for index, message in enumerate(tail_messages)
                    ],
                )

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(
            deps=ChatAgentDeps(db=db, user_id=user_id),
            message_history=message_history,
            on_complete=on_complete,
        )
        return adapter.streaming_response(event_stream)

    async def regenerate_from_user_message(  # noqa: C901
        self,
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
        try:
            run_input = AGUIAdapter.build_run_input(body)
        except ValidationError as e:
            return Response(content=e.json(), media_type='application/json', status_code=422)

        updates: dict[str, Any] = {}
        if not run_input.thread_id:
            updates['thread_id'] = conversation_id
        if not run_input.run_id:
            updates['run_id'] = uuid4_str()
        if updates:
            run_input = run_input.model_copy(update=updates)
        if run_input.thread_id != conversation_id:
            raise errors.RequestError(msg='请求体中的对话 ID 与路径不一致')

        try:
            forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        except ValidationError as e:
            raise errors.RequestError(msg=f'聊天扩展参数非法: {e.errors()[0]["msg"]}') from e
        if forwarded_props.mode != 'create':
            raise errors.RequestError(msg='当前聊天接口仅支持 create 模式')
        if forwarded_props.output_mode != AIChatOutputModeType.text:
            raise errors.RequestError(msg='当前聊天接口仅支持文本输出模式')
        if (
            forwarded_props.output_schema
            or forwarded_props.output_schema_name
            or forwarded_props.output_schema_description
        ):
            raise errors.RequestError(msg='当前聊天接口暂不支持结构化输出')

        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')

        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        if not message_rows:
            raise errors.RequestError(msg='对话消息不存在')
        target_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if target_index is None:
            raise errors.NotFoundError(msg='消息不存在')

        model_messages = ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows])
        target_message = model_messages[target_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持根据用户消息重生成')

        context_start_index = 0
        if conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1
        if target_index < context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

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

        message_history = model_messages[context_start_index : target_index + 1]
        preserved_history_count = target_index + 1
        preserved_context_count = len(message_history)

        async def on_complete(result: Any) -> None:
            persisted_messages = to_jsonable_python(list(result.all_messages()))
            assert isinstance(persisted_messages, list)

            await ai_conversation_dao.update(
                db,
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
            await ai_message_dao.delete_after_message_index(db, conversation_id, preserved_history_count)
            tail_messages = persisted_messages[preserved_context_count:]
            if tail_messages:
                await ai_message_dao.bulk_create(
                    db,
                    [
                        {
                            'conversation_id': conversation_id,
                            'provider_id': forwarded_props.provider_id,
                            'model_id': forwarded_props.model_id,
                            'message_index': preserved_history_count + index,
                            'message': message,
                        }
                        for index, message in enumerate(tail_messages)
                    ],
                )

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(
            deps=ChatAgentDeps(db=db, user_id=user_id),
            message_history=message_history,
            on_complete=on_complete,
        )
        return adapter.streaming_response(event_stream)

    async def regenerate_from_response_message(  # noqa: C901
        self,
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
        try:
            run_input = AGUIAdapter.build_run_input(body)
        except ValidationError as e:
            return Response(content=e.json(), media_type='application/json', status_code=422)

        updates: dict[str, Any] = {}
        if not run_input.thread_id:
            updates['thread_id'] = conversation_id
        if not run_input.run_id:
            updates['run_id'] = uuid4_str()
        if updates:
            run_input = run_input.model_copy(update=updates)

        try:
            forwarded_props = AIChatForwardedPropsParam.model_validate(run_input.forwarded_props or {})
        except ValidationError as e:
            raise errors.RequestError(msg=f'聊天扩展参数非法: {e.errors()[0]["msg"]}') from e
        if forwarded_props.mode != 'create':
            raise errors.RequestError(msg='当前聊天接口仅支持 create 模式')
        if forwarded_props.output_mode != AIChatOutputModeType.text:
            raise errors.RequestError(msg='当前聊天接口仅支持文本输出模式')
        if (
            forwarded_props.output_schema
            or forwarded_props.output_schema_name
            or forwarded_props.output_schema_description
        ):
            raise errors.RequestError(msg='当前聊天接口暂不支持结构化输出')

        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')

        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        if not message_rows:
            raise errors.RequestError(msg='对话消息不存在')
        target_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if target_index is None:
            raise errors.NotFoundError(msg='消息不存在')

        model_messages = ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows])
        if isinstance(model_messages[target_index], ModelRequest):
            raise errors.RequestError(msg='仅支持根据 AI 回复重生成')

        context_start_index = 0
        if conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1
        if target_index < context_start_index:
            raise errors.RequestError(msg='指定消息已不在当前上下文中')

        user_message_index = None
        for index in range(target_index - 1, context_start_index - 1, -1):
            if isinstance(model_messages[index], ModelRequest):
                user_message_index = index
                break
        if user_message_index is None:
            raise errors.RequestError(msg='未找到对应的用户消息')

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

        message_history = model_messages[context_start_index : user_message_index + 1]
        preserved_history_count = user_message_index + 1
        preserved_context_count = len(message_history)

        async def on_complete(result: Any) -> None:
            persisted_messages = to_jsonable_python(list(result.all_messages()))
            assert isinstance(persisted_messages, list)

            await ai_conversation_dao.update(
                db,
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
            await ai_message_dao.delete_after_message_index(db, conversation_id, preserved_history_count)
            tail_messages = persisted_messages[preserved_context_count:]
            if tail_messages:
                await ai_message_dao.bulk_create(
                    db,
                    [
                        {
                            'conversation_id': conversation_id,
                            'provider_id': forwarded_props.provider_id,
                            'model_id': forwarded_props.model_id,
                            'message_index': preserved_history_count + index,
                            'message': message,
                        }
                        for index, message in enumerate(tail_messages)
                    ],
                )

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(
            deps=ChatAgentDeps(db=db, user_id=user_id),
            message_history=message_history,
            on_complete=on_complete,
        )
        return adapter.streaming_response(event_stream)


ai_chat_service: ChatService = ChatService()
