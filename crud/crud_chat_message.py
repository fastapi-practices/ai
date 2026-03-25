from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIChatMessage


class CRUDAIChatMessage(CRUDPlus[AIChatMessage]):
    async def get_all(self, db: AsyncSession, conversation_id: str) -> Sequence[AIChatMessage]:
        """
        获取会话全部消息

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :return:
        """
        return await self.select_models_order(db, 'message_index', 'asc', conversation_id=conversation_id)

    async def get_select(self, conversation_id: str) -> Select:
        """
        获取会话消息查询表达式

        :param conversation_id: 会话 ID
        :return:
        """
        return await self.select_order('message_index', 'asc', conversation_id=conversation_id)

    async def bulk_create(self, db: AsyncSession, objs: list[dict[str, Any]]) -> None:
        """
        批量创建消息

        :param db: 数据库会话
        :param objs: 消息列表
        :return:
        """
        await self.bulk_create_models(db, objs)

    async def delete(self, db: AsyncSession, conversation_id: str) -> int:
        """
        删除会话全部消息

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :return:
        """
        return await self.delete_model_by_column(db, allow_multiple=True, conversation_id=conversation_id)


ai_chat_message_dao: CRUDAIChatMessage = CRUDAIChatMessage(AIChatMessage)
