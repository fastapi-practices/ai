from pydantic_ai import ModelRequest, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.context import ctx
from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.chat.generation.registry import get_generation_handler
from backend.plugin.ai.chat.session import AgentSession
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.dataclasses import ContextManagementPolicy
from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.policy.context import AIInvocationContext
from backend.plugin.ai.policy.registry import validate_ai_invocation
from backend.plugin.ai.protocol.base import ChatAgent, ChatModelMessage
from backend.plugin.ai.providers.registry import get_provider_adapter
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


def is_user_prompt_message(*, message: ChatModelMessage) -> bool:
    """
    判断是否为用户输入消息

    :param message: 模型消息
    :return:
    """
    return isinstance(message, ModelRequest) and bool(message.parts) and isinstance(message.parts[0], UserPromptPart)


async def open_chat_session(
    *,
    db: AsyncSession,
    forwarded_props: AIChatForwardedPropsParam,
    user_id: int | None = None,
    conversation_id: str | None = None,
) -> tuple[AgentSession, ChatAgent]:
    """
    解析供应商与模型，打开会话并构建代理

    :param db: 数据库会话
    :param forwarded_props: 聊天扩展参数
    :param user_id: 用户 ID
    :param conversation_id: 对话 ID
    :return:
    """
    provider = await ai_provider_dao.get(db, forwarded_props.provider_id)
    if not provider:
        raise errors.NotFoundError(msg='供应商不存在')
    if not provider.status:
        raise errors.RequestError(msg='此供应商暂不可用，请更换供应商或联系系统管理员')
    generation_handler = get_generation_handler(forwarded_props.generation_type)
    generation_handler.validate_provider_type(provider.type)
    model = await ai_model_dao.get_by_model_and_provider(db, forwarded_props.model_id, forwarded_props.provider_id)
    if not model:
        raise errors.NotFoundError(msg='供应商模型不存在')
    if not model.status:
        raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')
    invocation_context = None
    if user_id is not None:
        invocation_context = AIInvocationContext(
            provider_id=provider.id,
            provider_type=AIProviderType(provider.type),
            provider_name=provider.name,
            model_pk=model.id,
            model_id=model.model_id,
            user_id=user_id,
            is_superuser=ctx.is_superuser,
            mcp_ids=tuple(forwarded_props.mcp_ids or ()),
            generation_type=forwarded_props.generation_type,
            conversation_id=conversation_id,
        )
        await validate_ai_invocation(db=db, context=invocation_context)
    adapter = get_provider_adapter(provider.type)
    adapter.validate_model_id(model.model_id)
    session = await AgentSession.open(
        adapter=adapter,
        model_name=model.model_id,
        api_key=provider.api_key,
        base_url=provider.api_host,
    )
    session.invocation_context = invocation_context
    try:
        agent = await session.build_agent(
            db=db,
            forwarded_props=forwarded_props,
            generation_handler=generation_handler,
            context_management=ContextManagementPolicy(
                max_part_chars=model.context_max_part_chars,
                max_messages=model.context_max_messages,
                keep_messages=model.context_keep_messages,
                max_tokens=model.context_max_tokens,
                warning_threshold=settings.AI_CONTEXT_WARNING_THRESHOLD,
            ),
        )
    except ValueError as exc:
        await session.aclose()
        raise errors.RequestError(msg=f'模型配置无效: {exc}') from exc
    except Exception:
        await session.aclose()
        raise
    return session, agent
