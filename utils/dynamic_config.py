from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.dynamic_config import load_config


async def load_ai_config(db: AsyncSession) -> None:
    """
    获取 AI 配置

    :param db: 数据库会话
    :return:
    """
    mapping = {
        'AI_EXA_API_KEY': str,
        'AI_TAVILY_API_KEY': str,
    }
    await load_config(db, 'ai', mapping, 'AI_CONFIG_STATUS')
