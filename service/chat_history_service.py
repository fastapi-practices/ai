from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import cursor_paging_data
from backend.plugin.ai.crud.crud_chat_history import ai_chat_history_dao
from backend.plugin.ai.crud.crud_chat_message import ai_chat_message_dao
from backend.plugin.ai.model import AIChatHistory, AIChatMessage
from backend.plugin.ai.schema.chat import UpdateAIChatMessageParam
from backend.plugin.ai.schema.chat_history import (
    DeleteAIChatMessageResult,
    GetAIChatConversationDetail,
    UpdateAIChatConversationParam,
    UpdateAIChatConversationPinParam,
    UpdateAIChatHistoryParam,
)
from backend.plugin.ai.utils.message_parse import to_chat_messages
from backend.utils.timezone import timezone


class AIChatHistoryService:
    """AI 聊天历史服务"""

    @staticmethod
    async def get(*, db: AsyncSession, conversation_id: str, user_id: int) -> AIChatHistory:
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

    async def _get_history_and_messages(
        self, *, db: AsyncSession, conversation_id: str, user_id: int
    ) -> tuple[AIChatHistory, Sequence[AIChatMessage], list[ModelMessage] | Any]:
        """
        获取聊天会话、消息记录及模型消息

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history = await self.get(db=db, conversation_id=conversation_id, user_id=user_id)
        message_rows = await ai_chat_message_dao.get_all(db, chat_history.conversation_id)
        model_messages = (
            ModelMessagesTypeAdapter.validate_python([row.message for row in message_rows]) if message_rows else []
        )
        return chat_history, message_rows, model_messages

    async def get_detail(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> GetAIChatConversationDetail:
        """
        获取聊天历史详情

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        chat_history, message_rows, model_messages = await self._get_history_and_messages(
            db=db, conversation_id=conversation_id, user_id=user_id
        )
        message_ids = [row.id for row in message_rows]
        messages = to_chat_messages(
            model_messages,
            conversation_id=chat_history.conversation_id,
            message_ids=message_ids,
        )
        return GetAIChatConversationDetail(
            id=chat_history.id,
            conversation_id=chat_history.conversation_id,
            title=chat_history.title,
            provider_id=chat_history.provider_id,
            model_id=chat_history.model_id,
            is_pinned=chat_history.pinned_time is not None,
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
        chat_history = await self.get(db=db, conversation_id=conversation_id, user_id=user_id)
        title = ' '.join(obj.title.split())
        if not title:
            raise errors.RequestError(msg='会话标题不能为空')
        if len(title) > 256:
            raise errors.RequestError(msg='会话标题过长')
        return await ai_chat_history_dao.update_title(db, chat_history.id, title)

    async def update_pinned_status(
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
        chat_history = await self.get(db=db, conversation_id=conversation_id, user_id=user_id)
        return await ai_chat_history_dao.update_pinned_time(
            db, chat_history.id, timezone.now() if obj.is_pinned else None
        )

    async def get_editable_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
    ) -> tuple[AIChatHistory, str, list]:
        """
        获取可编辑消息及其前置历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :return:
        """
        chat_history, message_rows, model_messages = await self._get_history_and_messages(
            db=db, conversation_id=conversation_id, user_id=user_id
        )
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='聊天消息不存在')
        target_message = model_messages[message_row_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        first_part = target_message.parts[0]
        if not isinstance(first_part, UserPromptPart) or not isinstance(first_part.content, str):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        return chat_history, first_part.content, list(model_messages[:message_row_index])

    async def update_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
        obj: UpdateAIChatMessageParam,
    ) -> int:
        """
        更新指定聊天消息

        仅更新当前用户消息内容，不影响后续消息。

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :param obj: 更新参数
        :return:
        """
        _, message_rows, model_messages = await self._get_history_and_messages(
            db=db, conversation_id=conversation_id, user_id=user_id
        )
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='聊天消息不存在')
        target_message = model_messages[message_row_index]
        if not isinstance(target_message, ModelRequest):
            raise errors.RequestError(msg='仅支持编辑用户消息')
        first_part = target_message.parts[0]
        if not isinstance(first_part, UserPromptPart) or not isinstance(first_part.content, str):
            raise errors.RequestError(msg='仅支持编辑用户消息')

        content = ' '.join(obj.content.split())
        if not content:
            raise errors.RequestError(msg='消息内容不能为空')
        payload = deepcopy(message_rows[message_row_index].message)
        payload['parts'][0]['content'] = content
        return await ai_chat_message_dao.update(db, message_id, {'message': payload})

    async def get_regeneratable_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
    ) -> tuple[AIChatHistory, str, list]:
        """
        获取可重新生成的 AI 消息及其前置历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :return:
        """
        chat_history, message_rows, model_messages = await self._get_history_and_messages(
            db=db, conversation_id=conversation_id, user_id=user_id
        )
        message_row_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if message_row_index is None:
            raise errors.NotFoundError(msg='聊天消息不存在')
        target_message = model_messages[message_row_index]
        if not isinstance(target_message, ModelResponse):
            raise errors.RequestError(msg='仅支持重新生成 AI 消息')
        if message_row_index == 0:
            raise errors.RequestError(msg='缺少可用于重新生成的用户消息')
        previous_message = model_messages[message_row_index - 1]
        if not isinstance(previous_message, ModelRequest):
            raise errors.RequestError(msg='当前 AI 消息前缺少用户消息')
        previous_first_part = previous_message.parts[0]
        if not isinstance(previous_first_part, UserPromptPart) or not isinstance(previous_first_part.content, str):
            raise errors.RequestError(msg='当前 AI 消息前缺少用户消息')
        return chat_history, previous_first_part.content, list(model_messages[: message_row_index - 1])

    async def delete_message(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        message_id: int,
    ) -> DeleteAIChatMessageResult:
        """
        删除指定聊天消息

        删除时会先根据消息 ID 定位底层消息记录，
        再重建剩余消息记录及消息索引；若删除后会话已无任何消息，
        则连同话题记录一起删除。

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :param message_id: 消息 ID
        :return:
        """
        chat_history, message_rows, model_messages = await self._get_history_and_messages(
            db=db, conversation_id=conversation_id, user_id=user_id
        )
        target_message_index = next((index for index, row in enumerate(message_rows) if row.id == message_id), None)
        if target_message_index is None:
            raise errors.NotFoundError(msg='聊天消息不存在')
        remaining_messages = list(model_messages)
        del remaining_messages[target_message_index]
        if not remaining_messages:
            await ai_chat_message_dao.delete(db, conversation_id)
            await ai_chat_history_dao.delete(db, conversation_id, user_id)
            return DeleteAIChatMessageResult(deleted_conversation=True, remaining_message_count=0)

        remaining_message_rows = [row for row in message_rows if row.id != message_id]
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
        remaining_message_count = len(to_chat_messages(remaining_messages))
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
        await self.get(db=db, conversation_id=conversation_id, user_id=user_id)
        return await ai_chat_message_dao.delete(db, conversation_id)

    async def delete(self, *, db: AsyncSession, conversation_id: str, user_id: int) -> int:
        """
        删除聊天历史

        :param db: 数据库会话
        :param conversation_id: 会话 ID
        :param user_id: 用户 ID
        :return:
        """
        await self.get(db=db, conversation_id=conversation_id, user_id=user_id)
        await ai_chat_message_dao.delete(db, conversation_id)
        return await ai_chat_history_dao.delete(db, conversation_id, user_id)


ai_chat_history_service: AIChatHistoryService = AIChatHistoryService()
