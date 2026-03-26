from datetime import datetime

import sqlalchemy as sa

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIChatHistory
from backend.plugin.ai.schema.chat_history import CreateAIChatHistoryParam, UpdateAIChatHistoryParam


class CRUDAIChatHistory(CRUDPlus[AIChatHistory]):
    async def get(self, db: AsyncSession, pk: int) -> AIChatHistory | None:
        """
        获取聊天历史

        :param db: 数据库会话
        :param pk: 历史 ID
        :return:
        """
        return await self.select_model(db, pk)

    async def get_by_conversation_id(self, db: AsyncSession, conversation_id: str) -> AIChatHistory | None:
        """
        通过会话 ID 获取聊天历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :return:
        """
        return await self.select_model_by_column(db, conversation_id=conversation_id)

    async def get_select(self, user_id: int) -> Select[tuple[AIChatHistory]]:
        """
        获取聊天历史列表查询表达式

        :param user_id: 用户 ID
        :return:
        """
        stmt = select(self.model).where(self.model.user_id == user_id)
        pinned_first = sa.case((self.model.pinned_time.is_(None), 1), else_=0)
        return stmt.order_by(pinned_first, desc(self.model.id))

    async def create(self, db: AsyncSession, obj: CreateAIChatHistoryParam) -> None:
        """
        创建聊天历史

        :param db: 数据库会话
        :param obj: 创建聊天历史参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateAIChatHistoryParam) -> int:
        """
        更新聊天历史

        :param db: 数据库会话
        :param pk: 历史 ID
        :param obj: 更新聊天历史参数
        :return:
        """
        return await self.update_model(db, pk, obj)

    async def update_title(self, db: AsyncSession, pk: int, title: str) -> int:
        """
        更新聊天话题标题

        :param db: 数据库会话
        :param pk: 历史 ID
        :param title: 会话标题
        :return:
        """
        return await self.update_model(db, pk, {'title': title})

    async def update_pinned_time(self, db: AsyncSession, pk: int, pinned_time: datetime | None) -> int:
        """
        更新聊天话题置顶时间

        :param db: 数据库会话
        :param pk: 历史 ID
        :param pinned_time: 置顶时间
        :return:
        """
        return await self.update_model(db, pk, {'pinned_time': pinned_time})

    async def delete(self, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        通过会话 ID 删除聊天历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        return await self.delete_model_by_column(db, conversation_id=conversation_id, user_id=user_id)


ai_chat_history_dao: CRUDAIChatHistory = CRUDAIChatHistory(AIChatHistory)
