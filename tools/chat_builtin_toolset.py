from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from backend.plugin.ai.dataclasses import ChatAgentDeps


def build_chat_builtin_toolset() -> FunctionToolset[ChatAgentDeps]:
    """
    构建聊天内置工具集

    :return:
    """
    toolset = FunctionToolset[ChatAgentDeps]()

    @toolset.tool_plain
    def get_current_time() -> str:
        """
        获取当前时间

        :return:
        """
        from backend.utils.timezone import timezone

        return timezone.to_str(timezone.now())

    @toolset.tool
    async def list_my_quick_phrases(ctx: RunContext[ChatAgentDeps]) -> list[dict[str, int | str]]:
        """
        获取当前用户快捷短语列表

        :param ctx: 运行上下文
        :return:
        """
        from backend.plugin.ai.service.quick_phrase_service import ai_quick_phrase_service

        phrases = await ai_quick_phrase_service.get_all(db=ctx.deps.db, user_id=ctx.deps.user_id)
        return [{'id': item.id, 'title': item.title, 'content': item.content} for item in phrases]

    return toolset
