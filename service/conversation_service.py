from copy import deepcopy
from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import cursor_paging_data
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.schema.conversation import (
    ClearAIConversationContextResult,
    DeleteAIMessageResult,
    GetAIConversationDetail,
    UpdateAIConversationParam,
    UpdateAIConversationPinnedParam,
    UpdateAIConversationTitleParam,
)
from backend.plugin.ai.schema.message import UpdateAIMessageParam
from backend.plugin.ai.utils.message_parse import to_messages
from backend.utils.timezone import timezone


class AIConversationService:
    """AI 对话服务"""

    async def get(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> GetAIConversationDetail:
        """
        获取对话详情

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        message_rows = await ai_message_dao.get_all(db, conversation.conversation_id)
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        messages = to_messages(
            model_messages,
            conversation_id=conversation.conversation_id,
            message_ids=[row.id for row in message_rows],
        )
        return GetAIConversationDetail(
            id=conversation.id,
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            provider_id=conversation.provider_id,
            model_id=conversation.model_id,
            context_start_message_id=conversation.context_start_message_id,
            context_cleared_time=conversation.context_cleared_time,
            created_time=conversation.created_time,
            updated_time=conversation.updated_time,
            messages=messages,
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
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        title = ' '.join(obj.title.split())
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
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        return await ai_conversation_dao.update_pinned_time(
            db,
            conversation.id,
            timezone.now() if obj.is_pinned else None,
        )

    async def update_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
        obj: UpdateAIMessageParam,
    ) -> int:
        """
        编辑保存指定消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :param obj: 更新参数
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='消息不存在')
        target_message = model_messages[message_row_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not target_message.parts or not isinstance(target_message.parts[0], UserPromptPart):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        if not isinstance(target_message.parts[0].content, str):
            raise errors.RequestError(msg='当前消息暂不支持直接编辑')

        content = ' '.join(obj.content.split())
        if not content:
            raise errors.RequestError(msg='消息内容不能为空')
        payload = deepcopy(message_rows[message_row_index].message)
        payload['parts'][0]['content'] = content
        return await ai_message_dao.update(db, message_id, {'message': payload})

    async def delete_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
    ) -> DeleteAIMessageResult:
        """
        删除指定消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        target_message_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if target_message_index is None:
            raise errors.NotFoundError(msg='消息不存在')

        remaining_messages = list(model_messages)
        del remaining_messages[target_message_index]
        if not remaining_messages:
            await ai_message_dao.delete(db, conversation_id)
            await ai_conversation_dao.delete(db, conversation_id, user_id)
            return DeleteAIMessageResult(deleted_conversation=True, remaining_message_count=0)

        await ai_message_dao.delete_message(db, message_id)
        remaining_message_rows = [row for row in message_rows if row.id != message_id]
        for index, row in enumerate(remaining_message_rows):
            if row.message_index != index:
                await ai_message_dao.update(db, row.id, {'message_index': index})

        context_start_message_id = conversation.context_start_message_id
        if context_start_message_id == message_id:
            previous_rows = message_rows[:target_message_index]
            context_start_message_id = previous_rows[-1].id if previous_rows else None
        await ai_conversation_dao.update(
            db,
            conversation.id,
            UpdateAIConversationParam(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                provider_id=remaining_message_rows[-1].provider_id,
                model_id=remaining_message_rows[-1].model_id,
                user_id=conversation.user_id,
                pinned_time=conversation.pinned_time,
                context_start_message_id=context_start_message_id,
                context_cleared_time=conversation.context_cleared_time if context_start_message_id else None,
            ),
        )
        return DeleteAIMessageResult(
            deleted_conversation=False,
            remaining_message_count=len(to_messages(remaining_messages)),
        )

    async def clear_messages(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        清空对话消息

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        await ai_conversation_dao.update(
            db,
            conversation.id,
            UpdateAIConversationParam(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                provider_id=conversation.provider_id,
                model_id=conversation.model_id,
                user_id=conversation.user_id,
                pinned_time=conversation.pinned_time,
                context_start_message_id=None,
                context_cleared_time=None,
            ),
        )
        return await ai_message_dao.delete(db, conversation_id)

    async def clear_context(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
    ) -> ClearAIConversationContextResult:
        """
        清除对话上下文

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        message_rows = list(await ai_message_dao.get_all(db, conversation_id))
        context_start_message_id = message_rows[-1].id if message_rows else None
        context_cleared_time = timezone.now() if message_rows else None
        await ai_conversation_dao.update(
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
        return ClearAIConversationContextResult(
            context_start_message_id=context_start_message_id,
            context_cleared_time=context_cleared_time,
        )

    async def delete(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        删除对话

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        conversation = await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise errors.NotFoundError(msg='对话不存在')
        await ai_message_dao.delete(db, conversation_id)
        return await ai_conversation_dao.delete(db, conversation_id, user_id)


ai_conversation_service: AIConversationService = AIConversationService()
