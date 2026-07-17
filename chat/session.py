from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models import Model
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from backend.common.log import log
from backend.database.db import async_db_session
from backend.plugin.ai.chat.builder import build_model_settings
from backend.plugin.ai.chat.generation.base import GenerationHandler
from backend.plugin.ai.chat.persistence import (
    extract_assistant_messages,
    extract_assistant_run_messages,
    persist_completion,
    persist_terminal_completion,
)
from backend.plugin.ai.chat.pipeline import assemble_capabilities
from backend.plugin.ai.dataclasses import (
    ChatAgentDeps,
    ChatRunContext,
    CompletionPersistenceContext,
    ContextManagementPolicy,
)
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.policy.context import AIInvocationContext, AIInvocationResult
from backend.plugin.ai.policy.registry import notify_ai_invocation_result
from backend.plugin.ai.protocol.base import ChatAgent, ChatModelMessage, ChatProtocolAdapter
from backend.plugin.ai.providers.base import ProviderAdapter
from backend.plugin.ai.providers.http import build_retry_http_client
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


class AgentSession:
    """对话运行时会话，拥有 HTTP/SDK 客户端、模型与 agent 生命周期"""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter,
        model: Model,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self._http_client = http_client
        self._closed = False
        self.invocation_context: AIInvocationContext | None = None

    @classmethod
    async def open(
        cls,
        *,
        adapter: ProviderAdapter,
        model_name: str,
        api_key: str,
        base_url: str,
    ) -> 'AgentSession':
        """
        打开会话并构建模型实例

        :param adapter: 供应商适配器
        :param model_name: 模型名称
        :param api_key: API 密钥
        :param base_url: API 基础地址
        :return:
        """
        http_client = build_retry_http_client()
        try:
            model = adapter.create_model(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )
        except Exception:
            await http_client.aclose()
            raise
        return cls(adapter=adapter, model=model, http_client=http_client)

    async def aclose(self) -> None:
        """幂等关闭会话及其客户端资源"""
        if self._closed:
            return
        self._closed = True
        try:
            await self.adapter.aclose(self.model)
        finally:
            if not self._http_client.is_closed:
                await self._http_client.aclose()

    async def build_agent(
        self,
        *,
        db: AsyncSession,
        forwarded_props: AIChatForwardedPropsParam,
        generation_handler: GenerationHandler,
        context_management: ContextManagementPolicy,
    ) -> ChatAgent:
        """
        构建聊天代理

        :param db: 数据库会话
        :param forwarded_props: 聊天扩展参数
        :param generation_handler: 生成模式处理器
        :param context_management: 上下文管理策略
        :return:
        """
        profile = self.model.profile
        supports_tools = bool(profile.get('supports_tools', False))
        supported_native_tools = profile.get('supported_native_tools', frozenset())
        supports_image_output = bool(profile.get('supports_image_output', False))
        capabilities = await assemble_capabilities(
            db=db,
            adapter=self.adapter,
            forwarded_props=forwarded_props,
            supports_tools=supports_tools,
            supported_native_tools=supported_native_tools,
            supports_image_output=supports_image_output,
            context_management=context_management,
        )
        model_settings = build_model_settings(adapter=self.adapter, forwarded_props=forwarded_props)
        output_type = generation_handler.get_output_type()
        return Agent(
            name='fba-chat',
            deps_type=ChatAgentDeps,
            model=self.model,
            model_settings=model_settings,
            output_type=output_type,
            capabilities=capabilities,
        )

    def stream(
        self,
        *,
        user_id: int,
        agent: ChatAgent,
        run_context: ChatRunContext,
        protocol_adapter: ChatProtocolAdapter,
        accept: str | None,
        message_history: list[ChatModelMessage],
        persistence: CompletionPersistenceContext | None = None,
        on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]] | None = None,
        on_run_error: Callable[[str, list[ChatModelMessage]], Awaitable[None]] | None = None,
        on_interrupted: Callable[[list[ChatModelMessage]], Awaitable[None]] | None = None,
    ) -> StreamingResponse:
        """
        构建协议流式响应；on_finish 自动关闭会话

        :param user_id: 用户 ID
        :param agent: 聊天代理
        :param run_context: 协议运行上下文
        :param protocol_adapter: 协议适配器
        :param accept: Accept 请求头
        :param message_history: 消息历史
        :param persistence: 普通聊天持久化上下文
        :param on_complete: 自定义完成回调，未提供时默认调用 persist_completion
        :param on_run_error: 自定义异常回调，未提供时默认持久化错误终态
        :param on_interrupted: 自定义中断回调，未提供时默认持久化中断终态
        :return:
        """
        callbacks = (on_complete, on_run_error, on_interrupted)
        if persistence is None and any(callback is None for callback in callbacks):
            raise RuntimeError('缺少聊天生命周期回调')

        async def default_on_complete(result: AgentRunResult[Any]) -> None:
            assert persistence is not None
            async with async_db_session.begin() as db:
                await persist_completion(
                    db=db,
                    persistence=persistence,
                    messages=extract_assistant_run_messages(result),
                )

        async def default_on_run_error(message: str, messages: list[ChatModelMessage]) -> None:
            assert persistence is not None
            await persist_terminal_completion(
                persistence=persistence,
                messages=extract_assistant_messages(messages),
                status=AIMessageStatus.error,
                reason=message,
            )

        async def default_on_interrupted(messages: list[ChatModelMessage]) -> None:
            assert persistence is not None
            await persist_terminal_completion(
                persistence=persistence,
                messages=extract_assistant_messages(messages),
                status=AIMessageStatus.interrupted,
            )

        async def complete_with_policy(result: AgentRunResult[Any]) -> None:
            await (on_complete or default_on_complete)(result)
            if self.invocation_context is None:
                return
            async with async_db_session.begin() as db:
                await notify_ai_invocation_result(
                    db=db,
                    context=self.invocation_context,
                    result=AIInvocationResult.from_agent_result(result),
                )

        async def on_finish() -> None:
            try:
                await self.aclose()
            except Exception as exc:
                log.warning(f'关闭模型供应商客户端失败: {exc}')

        return protocol_adapter.build_streaming_response(
            user_id=user_id,
            agent=agent,
            run_context=run_context,
            accept=accept,
            message_history=message_history,
            on_complete=complete_with_policy,
            on_run_error=on_run_error or default_on_run_error,
            on_interrupted=on_interrupted or default_on_interrupted,
            on_finish=on_finish,
        )
