from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import Agent, AgentRunResult, BinaryImage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.builtin_tools import AbstractBuiltinTool, CodeExecutionTool, WebFetchTool
from pydantic_ai.capabilities import BuiltinTool, Thinking
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.exception import errors
from backend.common.log import log
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.dataclasses import ChatAgentDeps, ChatAgentParts, ChatCompletionPersistence
from backend.plugin.ai.enums import AIChatGenerationType, AIChatThinkingType, AIProviderType, AIWebSearchType
from backend.plugin.ai.model import AIModel, AIProvider
from backend.plugin.ai.protocol.base import ChatAgent, ChatModelMessage, ChatProtocolAdapter, ChatRunContext
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam
from backend.plugin.ai.schema.conversation import CreateAIConversationParam, UpdateAIConversationParam
from backend.plugin.ai.service.mcp_service import mcp_service
from backend.plugin.ai.tools.chat_builtin_toolset import build_chat_builtin_capability
from backend.plugin.ai.utils.capabilities.code_mode import build_code_mode_capability, should_enable_function_tools
from backend.plugin.ai.utils.capabilities.image_generation import build_image_generation_capability
from backend.plugin.ai.utils.capabilities.mcp import build_mcp_capability
from backend.plugin.ai.utils.capabilities.search import build_search_capabilities
from backend.plugin.ai.utils.chat_control import build_model_settings
from backend.plugin.ai.utils.conversation_control import normalize_generated_conversation_title
from backend.plugin.ai.utils.model_control import close_provider_model, get_provider_model


def is_user_prompt_message(*, message: ChatModelMessage) -> bool:
    """
    判断是否为用户输入消息

    :param message: 模型消息
    :return:
    """
    return isinstance(message, ModelRequest) and bool(message.parts) and isinstance(message.parts[0], UserPromptPart)


async def get_available_provider_model(
    *,
    db: AsyncSession,
    forwarded_props: AIChatForwardedPropsParam,
) -> tuple[AIProvider, AIModel]:
    """
    获取可用供应商及模型

    :param db: 数据库会话
    :param forwarded_props: 聊天扩展参数
    :return:
    """
    provider = await ai_provider_dao.get(db, forwarded_props.provider_id)
    if not provider:
        raise errors.NotFoundError(msg='供应商不存在')
    if not provider.status:
        raise errors.RequestError(msg='此供应商暂不可用，请更换供应商或联系系统管理员')
    if forwarded_props.generation_type == AIChatGenerationType.image and provider.type not in {
        AIProviderType.google,
        AIProviderType.openai_responses,
    }:
        raise errors.RequestError(msg='当前图片生成仅支持 Google 或 OpenAI Responses 供应商')

    model = await ai_model_dao.get_by_model_and_provider(db, forwarded_props.model_id, forwarded_props.provider_id)
    if not model:
        raise errors.NotFoundError(msg='供应商模型不存在')
    if not model.status:
        raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')
    if provider.type == AIProviderType.openrouter and '/' not in model.model_id:
        raise errors.RequestError(msg='OpenRouter 模型 ID 必须包含供应商前缀，例如 openai/gpt-4o-mini')
    return provider, model


async def build_chat_agent_parts(  # noqa: C901
    *,
    db: AsyncSession,
    forwarded_props: AIChatForwardedPropsParam,
    provider_type: int,
    supports_tools: bool,
    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]],
    supports_image_output: bool,
) -> ChatAgentParts:
    """
    构建聊天代理参数片段

    :param db: 数据库会话
    :param forwarded_props: 聊天扩展参数
    :param provider_type: 供应商类型
    :param supports_tools: 模型是否支持 function tools
    :param supported_builtin_tools: 模型支持的内置工具类型
    :param supports_image_output: 模型是否支持图片输出
    :return:
    """
    parts = ChatAgentParts()
    has_builtin_tools = False
    has_function_tool_sources = False

    if forwarded_props.thinking is not None:
        thinking_effort = (
            forwarded_props.thinking.value
            if isinstance(forwarded_props.thinking, AIChatThinkingType)
            else forwarded_props.thinking
        )
        parts.capabilities.append(Thinking(thinking_effort))

    if forwarded_props.mcp_ids:
        mcps = await mcp_service.get_by_ids(db=db, mcp_ids=forwarded_props.mcp_ids)
        parts.capabilities.extend(build_mcp_capability(mcp=mcp) for mcp in mcps)
        has_function_tool_sources = True

    auto_web_fetch = (
        AIProviderType(provider_type) != AIProviderType.google
        and forwarded_props.enable_builtin_tools
        and forwarded_props.generation_type == AIChatGenerationType.text
    )
    auto_web_fetch_enabled = (
        auto_web_fetch and forwarded_props.web_search != AIWebSearchType.off and WebFetchTool in supported_builtin_tools
    )
    parts.capabilities.extend(
        build_search_capabilities(
            web_search=forwarded_props.web_search,
            supported_builtin_tools=supported_builtin_tools,
            auto_web_fetch=auto_web_fetch,
        )
    )
    has_builtin_tools = (
        has_builtin_tools or forwarded_props.web_search == AIWebSearchType.builtin or auto_web_fetch_enabled
    )
    has_function_tool_sources = has_function_tool_sources or forwarded_props.web_search in {
        AIWebSearchType.exa,
        AIWebSearchType.tavily,
        AIWebSearchType.duckduckgo,
    }

    if (
        forwarded_props.enable_builtin_tools
        and forwarded_props.generation_type == AIChatGenerationType.text
        and CodeExecutionTool in supported_builtin_tools
    ):
        parts.capabilities.append(BuiltinTool(CodeExecutionTool()))
        has_builtin_tools = True

    if forwarded_props.generation_type == AIChatGenerationType.image:
        if not supports_image_output:
            raise errors.RequestError(msg='当前模型暂不支持图片生成，请更换模型')
        parts.capabilities.append(
            build_image_generation_capability(forwarded_props=forwarded_props, provider_type=provider_type)
        )
        has_builtin_tools = True

    function_tools_allowed = should_enable_function_tools(
        provider_type=provider_type,
        supports_tools=supports_tools,
        has_builtin_tools=has_builtin_tools,
    )
    if forwarded_props.enable_builtin_tools and function_tools_allowed:
        parts.capabilities.append(build_chat_builtin_capability())
        has_function_tool_sources = True

    if has_function_tool_sources and not function_tools_allowed:
        if AIProviderType(provider_type) == AIProviderType.google and has_builtin_tools:
            raise errors.RequestError(
                msg='Google 模型不支持同时使用内置工具和函数工具，请关闭 MCP 和本地搜索/关闭内置工具'
            )
        raise errors.RequestError(msg='当前模型不支持函数工具，请关闭 MCP、本地搜索或项目内置工具')
    if has_function_tool_sources:
        code_mode_capability = build_code_mode_capability(
            forwarded_props=forwarded_props,
            provider_type=provider_type,
            supports_tools=supports_tools,
            has_builtin_tools=has_builtin_tools,
        )
        if code_mode_capability is not None:
            parts.capabilities.append(code_mode_capability)
    return parts


async def build_chat_agent(*, db: AsyncSession, forwarded_props: AIChatForwardedPropsParam) -> Agent:
    """
    构建聊天代理

    :param db: 数据库会话
    :param forwarded_props: 聊天扩展参数
    :return:
    """
    provider, model = await get_available_provider_model(db=db, forwarded_props=forwarded_props)
    model_instance = get_provider_model(
        provider_type=provider.type,
        model_name=model.model_id,
        api_key=provider.api_key,
        base_url=provider.api_host,
    )
    try:
        profile = model_instance.profile
        model_settings = build_model_settings(chat_metadata=forwarded_props, provider_type=provider.type)
        agent_parts = await build_chat_agent_parts(
            db=db,
            forwarded_props=forwarded_props,
            provider_type=provider.type,
            supports_tools=profile.supports_tools,
            supported_builtin_tools=profile.supported_builtin_tools,
            supports_image_output=profile.supports_image_output,
        )
        return Agent(
            name='fba-chat',
            deps_type=ChatAgentDeps,
            model=model_instance,
            model_settings=model_settings,
            output_type=[BinaryImage, str] if forwarded_props.generation_type == AIChatGenerationType.image else str,
            capabilities=agent_parts.capabilities,
        )
    except ValueError as e:
        await close_provider_model(model_instance)
        raise errors.RequestError(msg=f'模型配置无效: {e}') from e
    except Exception:
        await close_provider_model(model_instance)
        raise


def prepare_run_context(
    *,
    conversation_id: str | None,
    forwarded_props: AIChatForwardedPropsParam,
    protocol_adapter: ChatProtocolAdapter,
    default_conversation_id: str | None = None,
    expected_conversation_id: str | None = None,
) -> ChatRunContext:
    """
    解析并补全运行参数

    :param conversation_id: 对话 ID
    :param forwarded_props: 聊天扩展参数
    :param protocol_adapter: 协议适配器
    :param default_conversation_id: 默认对话 ID
    :param expected_conversation_id: 期望对话 ID
    :return:
    """
    return protocol_adapter.build_run_context(
        conversation_id=conversation_id,
        forwarded_props=forwarded_props,
        default_conversation_id=default_conversation_id,
        expected_conversation_id=expected_conversation_id,
    )


async def persist_completion_messages(
    *,
    db: AsyncSession,
    persistence: ChatCompletionPersistence,
    messages: list[ChatModelMessage],
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

    payload_messages = to_jsonable_python(messages, by_alias=True)
    assert isinstance(payload_messages, list)
    current_conversation = persistence.conversation or await ai_conversation_dao.get_by_conversation_id(
        db, persistence.conversation_id
    )

    # 持久化对话
    normalized_title = normalize_generated_conversation_title(title=persistence.title)
    if current_conversation:
        await ai_conversation_dao.update(
            db,
            current_conversation.id,
            UpdateAIConversationParam(
                conversation_id=current_conversation.conversation_id,
                title=normalized_title,
                provider_id=persistence.forwarded_props.provider_id,
                model_id=persistence.forwarded_props.model_id,
                user_id=current_conversation.user_id,
                pinned_time=current_conversation.pinned_time,
                context_start_message_id=current_conversation.context_start_message_id,
                context_cleared_time=current_conversation.context_cleared_time,
            ),
        )
    else:
        await ai_conversation_dao.create(
            db,
            CreateAIConversationParam(
                conversation_id=persistence.conversation_id,
                title=normalized_title,
                provider_id=persistence.forwarded_props.provider_id,
                model_id=persistence.forwarded_props.model_id,
                user_id=persistence.user_id,
            ),
        )

    # 持久化消息
    insert_message_index = persistence.base_message_index
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
    elif persistence.insert_before_message_index is not None:
        await ai_message_dao.update_message_indexes_offset(
            db,
            persistence.conversation_id,
            persistence.insert_before_message_index,
            len(payload_messages),
        )

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


def stream_response(
    *,
    db: AsyncSession,
    user_id: int,
    agent: ChatAgent,
    run_context: ChatRunContext,
    protocol_adapter: ChatProtocolAdapter,
    accept: str | None,
    message_history: list[ChatModelMessage],
    on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]],
    persistence: ChatCompletionPersistence,
) -> StreamingResponse:
    """
    流式响应

    :param db: 数据库会话
    :param user_id: 用户 ID
    :param agent: 聊天代理
    :param run_context: 运行上下文
    :param protocol_adapter: 协议适配器
    :param accept: Accept 请求头
    :param message_history: 消息历史
    :param on_complete: 完成回调
    :param persistence: 持久化上下文
    :return:
    """

    async def handle_run_error(message: str) -> None:
        raw_error_message = ' '.join(message.split()) if message else ''
        try:
            error_message = raw_error_message or '模型请求失败，请稍后重试'
            await persist_completion_messages(
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
            log_message = (
                f'聊天运行失败，已写入对话记录 conversation_id={persistence.conversation_id}: {raw_error_message}'
            )
            log.warning(log_message)

    async def handle_finish() -> None:
        try:
            await close_provider_model(agent.model)
        except Exception as e:
            log.warning(f'关闭模型供应商客户端失败: {e}')

    return protocol_adapter.build_streaming_response(
        db=db,
        user_id=user_id,
        agent=agent,
        run_context=run_context,
        accept=accept,
        message_history=message_history,
        on_complete=on_complete,
        on_run_error=handle_run_error,
        on_finish=handle_finish,
    )
