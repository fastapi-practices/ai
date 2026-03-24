from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.plugin.ai.crud.crud_chat_history import ai_chat_history_dao
from backend.plugin.ai.model import AIChatHistory
from backend.plugin.ai.schema.chat import GetAIChatMessageDetail
from backend.plugin.ai.schema.chat_history import (
    DeleteAIChatMessageResult,
    GetAIChatConversationDetail,
    GetAIChatConversationItem,
    GetAIChatConversationList,
    UpdateAIChatConversationParam,
    UpdateAIChatConversationPinParam,
    UpdateAIChatHistoryParam,
)
from backend.plugin.ai.utils.message_parse import (
    build_chat_transcript,
    delete_model_message_by_index,
    get_chat_transcript_item,
    parse_model_messages,
    serialize_model_messages,
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
    async def get_recent_list(
        *,
        db: AsyncSession,
        user_id: int,
        limit: int,
        before: datetime | None = None,
    ) -> GetAIChatConversationList:
        """
        获取最近聊天历史列表

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param limit: 返回数量
        :param before: 查询游标
        :return:
        """
        chat_histories = list(await ai_chat_history_dao.get_recent_list(db, user_id, limit, before))
        has_more = len(chat_histories) > limit
        visible_chat_histories = chat_histories[:limit]
        items = []
        for chat_history in visible_chat_histories:
            last_activity_time = chat_history.updated_time or chat_history.created_time
            transcript = build_chat_transcript(
                parse_model_messages(chat_history.messages),
                conversation_id=chat_history.conversation_id,
            )
            last_message = transcript[-1].content if transcript else None
            items.append(
                GetAIChatConversationItem(
                    id=chat_history.id,
                    conversation_id=chat_history.conversation_id,
                    title=chat_history.title,
                    provider_id=chat_history.provider_id,
                    model_id=chat_history.model_id,
                    user_id=chat_history.user_id,
                    is_pinned=chat_history.pinned_time is not None,
                    pinned_time=chat_history.pinned_time,
                    last_message=last_message,
                    message_count=len(transcript),
                    last_activity_time=last_activity_time,
                    created_time=chat_history.created_time,
                    updated_time=chat_history.updated_time,
                )
            )
        next_before = None
        if has_more and visible_chat_histories:
            next_before = visible_chat_histories[-1].updated_time or visible_chat_histories[-1].created_time
        return GetAIChatConversationList(items=items, has_more=has_more, next_before=next_before)

    async def get_detail(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> GetAIChatConversationDetail:
        """
        获取聊天历史详情

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history = await self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        model_messages = parse_model_messages(chat_history.messages)
        messages = [
            GetAIChatMessageDetail.model_validate(message)
            for message in to_chat_messages(model_messages, conversation_id=chat_history.conversation_id)
        ]
        last_activity_time = chat_history.updated_time or chat_history.created_time
        return GetAIChatConversationDetail(
            id=chat_history.id,
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            is_pinned=chat_history.pinned_time is not None,
            pinned_time=chat_history.pinned_time,
            last_message=messages[-1].content if messages else None,
            message_count=len(messages),
            last_activity_time=last_activity_time,
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
            messages=chat_history.messages or [],
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
            messages=chat_history.messages or [],
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
        model_messages = parse_model_messages(chat_history.messages)
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
        model_messages = parse_model_messages(chat_history.messages)
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
        model_messages = parse_model_messages(chat_history.messages)
        remaining_messages = delete_model_message_by_index(
            model_messages,
            message_index=message_index,
            conversation_id=chat_history.conversation_id,
        )
        if not remaining_messages:
            await ai_chat_history_dao.delete_by_conversation_id(db, conversation_id, user_id)
            return DeleteAIChatMessageResult(deleted_conversation=True, remaining_message_count=0)

        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            pinned_time=chat_history.pinned_time,
            messages=serialize_model_messages(remaining_messages),
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
        payload = UpdateAIChatHistoryParam(
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            user_id=chat_history.user_id,
            pinned_time=chat_history.pinned_time,
            messages=[],
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
        return await ai_chat_history_dao.delete_by_conversation_id(db, conversation_id, user_id)


ai_chat_history_service: AIChatHistoryService = AIChatHistoryService()
