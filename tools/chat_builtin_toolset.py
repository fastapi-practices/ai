from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from backend.database.db import async_db_session
from backend.plugin.ai.dataclasses import ChatAgentDeps
from backend.utils.timezone import timezone


def build_chat_builtin_capability() -> AbstractCapability[ChatAgentDeps]:
    """构建聊天内置工具能力"""
    toolset: FunctionToolset[ChatAgentDeps] = FunctionToolset[ChatAgentDeps]()

    @toolset.tool_plain(defer_loading=True)
    def get_current_time() -> str:
        """获取当前时间"""
        return timezone.to_str(timezone.now())

    @toolset.tool(defer_loading=True)
    async def get_my_quick_phrases(ctx: RunContext[ChatAgentDeps]) -> list[dict[str, int | str]]:
        """
        获取当前用户快捷短语列表

        :param ctx: 运行上下文
        :return:
        """
        from backend.plugin.ai.service.quick_phrase_service import ai_quick_phrase_service

        async with async_db_session() as db:
            phrases = await ai_quick_phrase_service.get_all(db=db, user_id=ctx.deps.user_id)
        return [{'id': item.id, 'title': item.title, 'content': item.content} for item in phrases]

    return Toolset(toolset)
