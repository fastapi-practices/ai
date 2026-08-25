import anyio

from pydantic_ai import ModelRequest, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.plugin.ai.chat.session import AgentSession
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
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
    model = await ai_model_dao.get_by_model_and_provider(db, forwarded_props.model_id, forwarded_props.provider_id)
    if not model:
        raise errors.NotFoundError(msg='供应商模型不存在')
    if not model.status:
        raise errors.RequestError(msg='此模型暂不可用，请更换模型或联系系统管理员')

    adapter = get_provider_adapter(provider.type)
    adapter.validate_model_id(model.model_id)
    session = await AgentSession.open(
        adapter=adapter,
        model_name=model.model_id,
        api_key=provider.api_key,
        base_url=provider.api_host,
    )
    try:
        agent = await session.build_agent()
    except ValueError as exc:
        # 屏蔽取消：任务取消时仍完成客户端关闭，避免连接泄漏
        with anyio.CancelScope(shield=True):
            await session.aclose()
        raise errors.RequestError(msg=f'模型配置无效: {exc}') from exc
    except BaseException:
        with anyio.CancelScope(shield=True):
            await session.aclose()
        raise
    return session, agent
