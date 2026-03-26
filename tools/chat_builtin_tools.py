from typing import Any

from pydantic_ai import Agent, RunContext

from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.service.quick_phrase_service import ai_quick_phrase_service
from backend.utils.timezone import timezone


def register_chat_builtin_tools(agent: Agent) -> None:
    """注册聊天内置工具"""

    @agent.tool
    def get_current_time(_: RunContext[Any]) -> str:
        """获取当前时间"""
        return timezone.now().isoformat()

    @agent.tool
    async def list_my_quick_phrases(ctx: RunContext[Any]) -> list[dict[str, Any]]:
        """获取当前用户快捷短语列表"""
        phrases = await ai_quick_phrase_service.get_all(db=ctx.deps.db, user_id=ctx.deps.user_id)
        return [{'id': item.id, 'title': item.title, 'content': item.content} for item in phrases]

    @agent.tool
    async def list_provider_models(ctx: RunContext[Any], provider_id: int) -> list[str]:
        """获取指定供应商可用模型 ID"""
        models = await ai_model_dao.get_all(ctx.deps.db, provider_id=provider_id)
        return [item.model_id for item in models if item.status]
