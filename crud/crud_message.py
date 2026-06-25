from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIMessage
from backend.utils.timezone import timezone


class CRUDAIMessage(CRUDPlus[AIMessage]):
    """AI 消息数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIMessage | None:
        """
        获取消息

        :param db: 数据库会话
        :param pk: ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_all(self, db: AsyncSession, conversation_id: str) -> Sequence[AIMessage]:
        """
        获取对话全部消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :return:
        """
        stmt = (
            select(self.model)
            .where(self.model.conversation_id == conversation_id, self.model.deleted == 0)
            .order_by(self.model.message_index.asc(), self.model.id.asc())
        )
        return (await db.scalars(stmt)).all()

    async def get_select(self, conversation_id: str) -> Select:
        """
        获取对话消息查询表达式

        :param conversation_id: 对话 ID
        :return:
        """
        return (
            select(self.model)
            .where(self.model.conversation_id == conversation_id, self.model.deleted == 0)
            .order_by(self.model.message_index.asc(), self.model.id.asc())
        )

    async def get_next_message_index(self, db: AsyncSession, conversation_id: str) -> int:
        """
        获取下一条消息索引

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :return:
        """
        stmt = select(func.max(self.model.message_index)).where(
            self.model.conversation_id == conversation_id,
            self.model.deleted == 0,
        )
        max_message_index = await db.scalar(stmt)
        return (max_message_index if max_message_index is not None else -1) + 1

    async def bulk_create(self, db: AsyncSession, objs: list[dict[str, Any]]) -> None:
        """
        批量创建消息

        :param db: 数据库会话
        :param objs: 消息列表
        :return:
        """
        now = timezone.now()
        payloads = [
            {
                **obj,
                'created_time': obj.get('created_time', now),
                'updated_time': obj.get('updated_time'),
            }
            for obj in objs
        ]
        await self.bulk_create_models(db, payloads)

    async def update_message_indexes_offset(
        self,
        db: AsyncSession,
        conversation_id: str,
        start_message_index: int,
        offset: int,
    ) -> int:
        """
        平移指定索引及之后的消息索引

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param start_message_index: 起始消息索引
        :param offset: 平移偏移量
        :return:
        """
        messages = await self.select_models_order(
            db,
            'message_index',
            'asc',
            conversation_id=conversation_id,
            message_index__ge=start_message_index,
            deleted=0,
        )

        if not messages:
            return 0

        return await self.bulk_update_models(
            db,
            [
                {
                    'id': message.id,
                    'message_index': message.message_index + offset,
                }
                for message in messages
            ],
        )

    async def update(self, db: AsyncSession, pk: int, obj: dict[str, Any]) -> int:
        """
        更新消息

        :param db: 数据库会话
        :param pk: 消息 ID
        :param obj: 更新内容
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete_message_index_range(
        self,
        db: AsyncSession,
        conversation_id: str,
        start_message_index: int,
        end_message_index: int,
    ) -> int:
        """
        按消息索引范围逻辑删除消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param start_message_index: 起始消息索引
        :param end_message_index: 结束消息索引
        :return:
        """
        messages = await self.select_models_order(
            db,
            'message_index',
            'asc',
            conversation_id=conversation_id,
            message_index__ge=start_message_index,
            message_index__le=end_message_index,
            deleted=0,
        )
        if not messages:
            return 0
        deleted_time = timezone.now()
        return await self.bulk_update_models(
            db,
            [
                {
                    'id': message.id,
                    'deleted': message.id,
                    'deleted_time': deleted_time,
                }
                for message in messages
            ],
        )

    async def delete_message(self, db: AsyncSession, pk: int) -> int:
        """
        删除指定消息

        :param db: 数据库会话
        :param pk: 消息 ID
        :return:
        """
        return await self.delete_model_by_column(
            db,
            logical_deletion=True,
            deleted_flag_column='deleted',
            deleted_flag_value=self.model.id,
            deleted_at_column='deleted_time',
            deleted_at_factory=timezone.now(),
            id=pk,
            deleted=0,
        )

    async def delete(self, db: AsyncSession, conversation_id: str) -> int:
        """
        删除对话全部消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :return:
        """
        return await self.delete_model_by_column(
            db,
            allow_multiple=True,
            logical_deletion=True,
            deleted_flag_column='deleted',
            deleted_flag_value=self.model.id,
            deleted_at_column='deleted_time',
            deleted_at_factory=timezone.now(),
            conversation_id=conversation_id,
            deleted=0,
        )


ai_message_dao: CRUDAIMessage = CRUDAIMessage(AIMessage)
