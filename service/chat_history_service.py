from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import cursor_paging_data
from backend.plugin.ai.crud.crud_chat_history import ai_chat_history_dao
from backend.plugin.ai.crud.crud_chat_message import ai_chat_message_dao
from backend.plugin.ai.model import AIChatHistory
from backend.plugin.ai.schema.chat import GetAIChatMessageDetail
from backend.plugin.ai.schema.chat_history import (
    DeleteAIChatMessageResult,
    GetAIChatConversationDetail,
    UpdateAIChatConversationParam,
    UpdateAIChatConversationPinParam,
    UpdateAIChatHistoryParam,
)
from backend.plugin.ai.utils.message_parse import (
    build_chat_transcript,
    delete_model_message_by_index,
    get_chat_transcript_item,
    parse_model_messages,
    to_chat_messages,
    truncate_model_messages_by_index,
)
from backend.utils.timezone import timezone


class AIChatHistoryService:
    """AI 聊天历史服务"""

    @staticmethod
    async def get_conversation(*, db: AsyncSession, conversation_id: str, user_id: int) -> AIChatHistory:
        """
        获取聊天会话

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history = await ai_chat_history_dao.get_by_conversation_id(db, conversation_id)
        if not chat_history:
            raise errors.NotFoundError(msg='聊天历史不存在')
        if chat_history.user_id != user_id:
            raise errors.NotFoundError(msg='聊天历史不存在')
        return chat_history

    @staticmethod
    async def get_list(
        *,
        db: AsyncSession,
        user_id: int,
    ) -> dict[str, Any]:
        """
        获取聊天历史列表

        :param db: 数据库会话
        :param user_id: 用户 ID
        :return:
        """
        chat_history_select = await ai_chat_history_dao.get_select(user_id)
        page_data = await cursor_paging_data(db, chat_history_select)
        page_data['items'] = [
            {
                'id': item.id,
                'conversation_id': item.conversation_id,
                'title': item.title,
                'is_pinned': item.pinned_time is not None,
                'created_time': item.created_time,
                'updated_time': item.updated_time,
            }
            for item in page_data['items']
        ]
        return page_data

    async def get_detail(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> GetAIChatConversationDetail:
        """
        获取聊天历史详情

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        message_rows = await ai_chat_message_dao.get_all(db, chat_history.conversation_id)
        model_messages = parse_model_messages([row.message for row in message_rows])
        messages = [
            GetAIChatMessageDetail.model_validate(message)
            for message in to_chat_messages(model_messages, conversation_id=chat_history.conversation_id)
        ]
        return GetAIChatConversationDetail(
            id=chat_history.id,
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            is_pinned=chat_history.pinned_time is not None,
            message_count=len(messages),
            created_time=chat_history.created_time,
            updated_time=chat_history.updated_time,
            messages=messages,
        )

    async def update(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        obj: UpdateAIChatConversationParam,
    ) -> int:
        """
        更新聊天话题

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param obj: 更新参数
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        title = ' '.join(obj.title.split())
        if not title:
            raise errors.RequestError(msg='会话标题不能为空')
        if len(title) > 256:
            raise errors.RequestError(msg='会话标题过长')
        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            pinned_time=chat_history.pinned_time,
        )
        return await ai_chat_history_dao.update(db, chat_history.id, payload)

    async def update_pin(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        obj: UpdateAIChatConversationPinParam,
    ) -> int:
        """
        更新聊天话题置顶状态

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param obj: 更新参数
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            pinned_time=timezone.now() if obj.is_pinned else None,
        )
        return await ai_chat_history_dao.update(db, chat_history.id, payload)

    async def get_editable_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_index: int,
    ) -> tuple[AIChatHistory, str, list]:
        """
        获取可编辑消息及其前置历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_index: 消息索引
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        message_rows = await ai_chat_message_dao.get_all(db, chat_history.conversation_id)
        model_messages = parse_model_messages([row.message for row in message_rows])
        target_item = get_chat_transcript_item(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        if target_item.role != 'user':
            raise errors.RequestError(msg='仅支持编辑用户消息')
        truncated_messages = truncate_model_messages_by_index(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        return chat_history, target_item.content, truncated_messages

    async def get_regeneratable_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_index: int,
    ) -> tuple[AIChatHistory, str, list]:
        """
        获取可重新生成的 AI 消息及其前置历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_index: 消息索引
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        message_rows = await ai_chat_message_dao.get_all(db, chat_history.conversation_id)
        model_messages = parse_model_messages([row.message for row in message_rows])
        target_item = get_chat_transcript_item(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        if target_item.role != 'model':
            raise errors.RequestError(msg='仅支持重新生成 AI 消息')
        if message_index == 0:
            raise errors.RequestError(msg='缺少可用于重新生成的用户消息')

        previous_item = get_chat_transcript_item(
            model_messages,
            message_index=message_index - 1,
            conversation_id=chat_history.conversation_id,
        )
        if previous_item.role != 'user':
            raise errors.RequestError(msg='当前 AI 消息前缺少用户消息')
        truncated_messages = truncate_model_messages_by_index(
            model_messages,
            message_index=previous_item.message_index,
            conversation_id=chat_history.conversation_id,
        )
        return chat_history, previous_item.content, truncated_messages

    async def delete_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_index: int,
    ) -> DeleteAIChatMessageResult:
        """
        删除指定聊天消息

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_index: 消息索引
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        message_rows = await ai_chat_message_dao.get_all(db, chat_history.conversation_id)
        model_messages = parse_model_messages([row.message for row in message_rows])
        target_item = get_chat_transcript_item(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        remaining_messages = delete_model_message_by_index(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        if not remaining_messages:
            await ai_chat_message_dao.delete(db, conversation_id)
            await ai_chat_history_dao.delete(db, conversation_id, user_id)
            return DeleteAIChatMessageResult(deleted_conversation=True, remaining_message_count=0)

        remaining_message_rows = [
            row
            for row in message_rows
            if row.message_index != target_item.model_message_index
        ]
        await ai_chat_message_dao.delete(db, conversation_id)
        await ai_chat_message_dao.bulk_create(
            db,
            [
                {
                    'conversation_id': conversation_id,
                    'provider_id': row.provider_id,
                    'model_id': row.model_id,
                    'message_index': index,
                    'message': row.message,
                }
                for index, row in enumerate(remaining_message_rows)
            ],
        )
        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=remaining_message_rows[-1].provider_id,
            model_id=remaining_message_rows[-1].model_id,
            user_id=chat_history.user_id,
            pinned_time=chat_history.pinned_time,
        )
        await ai_chat_history_dao.update(db, chat_history.id, payload)
        remaining_message_count = len(
            build_chat_transcript(remaining_messages, conversation_id=chat_history.conversation_id)
        )
        return DeleteAIChatMessageResult(
            deleted_conversation=False,
            remaining_message_count=remaining_message_count,
        )

    async def clear_messages(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        清空聊天消息历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        await ai_chat_message_dao.delete(db, conversation_id)
        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            pinned_time=chat_history.pinned_time,
        )
        return await ai_chat_history_dao.update(db, chat_history.id, payload)

    async def delete(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        删除聊天历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        await ai_chat_message_dao.delete(db, conversation_id)
        return await ai_chat_history_dao.delete(db, conversation_id, user_id)


ai_chat_history_service: AIChatHistoryService = AIChatHistoryService()
