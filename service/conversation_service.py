from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import cursor_paging_data
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import ChatConversationState
from backend.plugin.ai.model.conversation import AIConversation
from backend.plugin.ai.protocol.ag_ui.snapshot_builder import serialize_messages_to_snapshot
from backend.plugin.ai.schema.conversation import (
    GetAIConversationDetail,
    UpdateAIConversationParam,
    UpdateAIConversationPinnedParam,
    UpdateAIConversationTitleParam,
)
from backend.plugin.ai.utils.conversation_control import normalize_conversation_title
from backend.utils.timezone import timezone


class AIConversationService:
    """AI 对话服务"""

    @staticmethod
    async def get_owned_conversation(
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        must_exist: bool = True,
    ) -> AIConversation | None:
        """
        获取当前用户所属对话

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param must_exist: 对话是否必须存在
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation:
            if must_exist:
                raise errors.NotFoundError(msg='对话不存在')
            return None
        if conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        return conversation

    async def get_chat_state(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        must_exist: bool,
        require_messages: bool = False,
    ) -> ChatConversationState:
        """
        加载聊天上下文状态

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param must_exist: 对话是否必须存在
        :param require_messages: 是否要求对话消息存在
        :return:
        """
        conversation = await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            must_exist=must_exist,
        )
        if not conversation:
            return ChatConversationState(
                conversation=None,
                message_rows=[],
                model_messages=[],
                context_start_index=0,
            )
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        if require_messages and not message_rows:
            raise errors.RequestError(msg='对话消息不存在')
        context_start_index = 0
        if conversation.context_start_message_id is not None:
            boundary_index = next(
                (index for index, row in enumerate(message_rows) if row.id == conversation.context_start_message_id),
                None,
            )
            if boundary_index is not None:
                context_start_index = boundary_index + 1
        return ChatConversationState(
            conversation=conversation,
            message_rows=message_rows,
            model_messages=(
                list(ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]))
                if message_rows
                else []
            ),
            context_start_index=context_start_index,
        )

    async def get(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> GetAIConversationDetail:
        """
        获取对话详情

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        message_rows = await ai_message_dao.get_all(db, conversation.conversation_id)
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        messages_snapshot = serialize_messages_to_snapshot(
            model_messages,
            conversation_id=conversation.conversation_id,
            message_ids=[row.id for row in message_rows],
            provider_ids=[row.provider_id for row in message_rows],
            model_ids=[row.model_id for row in message_rows],
        )
        return GetAIConversationDetail(
            id=conversation.id,
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            is_pinned=conversation.pinned_time is not None,
            provider_id=conversation.provider_id,
            model_id=conversation.model_id,
            context_start_message_id=conversation.context_start_message_id,
            context_cleared_time=conversation.context_cleared_time,
            created_time=conversation.created_time,
            updated_time=conversation.updated_time,
            messages_snapshot=messages_snapshot,
        )

    @staticmethod
    async def get_list(*, db: AsyncSession, user_id: int) -> dict[str, Any]:
        """
        获取对话列表

        :param db: 数据库会话
        :param user_id: 用户 ID
        :return:
        """
        conversation_select = await ai_conversation_dao.get_select(user_id)
        page_data = await cursor_paging_data(db, conversation_select)
        page_data['items'] = [
            {
                'id': item['id'],
                'conversation_id': item['conversation_id'],
                'title': item['title'],
                'is_pinned': item['pinned_time'] is not None,
                'created_time': item['created_time'],
                'updated_time': item['updated_time'],
            }
            for item in page_data['items']
        ]
        return page_data

    async def update(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        obj: UpdateAIConversationTitleParam,
    ) -> int:
        """
        更新对话标题

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param obj: 更新参数
        :return:
        """
        conversation = await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        title = normalize_conversation_title(title=obj.title, fallback='')
        if not title:
            raise errors.RequestError(msg='对话标题不能为空')
        if len(title) > 256:
            raise errors.RequestError(msg='对话标题过长')
        return await ai_conversation_dao.update_title(db, conversation.id, title)

    async def update_pinned_status(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        obj: UpdateAIConversationPinnedParam,
    ) -> int:
        """
        更新对话置顶状态

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param obj: 更新参数
        :return:
        """
        conversation = await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return await ai_conversation_dao.update_pinned_time(
            db,
            conversation.id,
            timezone.now() if obj.is_pinned else None,
        )

    async def clear_context(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        清除对话上下文

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        context_start_message_id = message_rows[-1].id if message_rows else None
        context_cleared_time = timezone.now() if message_rows else None
        return await ai_conversation_dao.update(
            db,
            conversation.id,
            UpdateAIConversationParam(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                provider_id=conversation.provider_id,
                model_id=conversation.model_id,
                user_id=conversation.user_id,
                pinned_time=conversation.pinned_time,
                context_start_message_id=context_start_message_id,
                context_cleared_time=context_cleared_time,
            ),
        )

    async def delete(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        删除对话

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        await ai_message_dao.delete(db, conversation_id)
        return await ai_conversation_dao.delete(db, conversation_id, user_id)


ai_conversation_service: AIConversationService = AIConversationService()
