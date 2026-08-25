from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.common.pagination import cursor_paging_data
from backend.database.db import async_db_session
from backend.plugin.ai.chat.runs import has_active_run, request_stop_run, wait_run_stopped
from backend.plugin.ai.crud.crud_conversation import ai_conversation_dao
from backend.plugin.ai.crud.crud_message import ai_message_dao
from backend.plugin.ai.dataclasses import ChatConversationState
from backend.plugin.ai.enums import AIMessageStatus
from backend.plugin.ai.model.conversation import AIConversation
from backend.plugin.ai.protocol.registry import get_chat_protocol_adapter
from backend.plugin.ai.schema.conversation import (
    GetAIConversationDetail,
    UpdateAIConversationPinnedParam,
    UpdateAIConversationTitleParam,
)
from backend.plugin.ai.utils.conversation_control import normalize_conversation_title
from backend.plugin.ai.utils.message_storage import expand_message_row_metadata, expand_message_rows
from backend.utils.timezone import timezone


class AIConversationService:
    """AI 对话服务"""

    @staticmethod
    async def ensure_idle(*, db: AsyncSession, conversation_id: str) -> None:
        """
        确认对话可以开始新的生成

        进行中的后台任务会拒绝新请求。无后台任务的残留 pending 会被标为中断。

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :return:
        """
        if await has_active_run(conversation_id):
            raise errors.ConflictError(msg='当前对话正在生成，请稍后再试')
        pending_rows = await ai_message_dao.get_pending(db, conversation_id)
        for row in pending_rows:
            await ai_message_dao.update_pending(
                db,
                row.id,
                {'status': AIMessageStatus.interrupted},
            )

    @staticmethod
    async def reconcile_stale_pending(*, stale_after_seconds: int = 60) -> int:
        """
        将租约已失效的残留 pending 消息收敛为中断

        :param stale_after_seconds: 最短静默秒数
        :return:
        """
        stale_before = timezone.now() - timedelta(seconds=stale_after_seconds)
        async with async_db_session.begin() as db:
            stale_rows = await ai_message_dao.get_stale_pending(db, stale_before)
            rows_by_conversation: dict[str, list[int]] = {}
            for row in stale_rows:
                rows_by_conversation.setdefault(row.conversation_id, []).append(row.id)

        reconciled = 0
        for conversation_id, message_ids in rows_by_conversation.items():
            try:
                if await has_active_run(conversation_id):
                    continue
                async with async_db_session.begin() as db:
                    for message_id in message_ids:
                        reconciled += await ai_message_dao.update_pending(
                            db,
                            message_id,
                            {'status': AIMessageStatus.interrupted},
                        )
            except Exception as exc:
                log.warning(f'收敛残留聊天消息失败 conversation_id={conversation_id}: {exc}')
        return reconciled

    async def stop_generation(
        self,
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
    ) -> None:
        """
        停止对话当前生成

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :return:
        """
        await self.get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            for_update=True,
        )
        if await request_stop_run(conversation_id):
            if not await wait_run_stopped(conversation_id):
                raise errors.ServerError(msg='停止对话生成超时，请稍后重试')
            return
        await self.ensure_idle(db=db, conversation_id=conversation_id)

    @staticmethod
    async def get_owned_conversation(
        *,
        db: AsyncSession,
        conversation_id: str,
        user_id: int,
        must_exist: bool = True,
        for_update: bool = False,
    ) -> AIConversation | None:
        """
        获取当前用户所属对话

        :param db: 数据库会话
        :param conversation_id: 对话 ID
        :param user_id: 用户 ID
        :param must_exist: 对话是否必须存在
        :param for_update: 是否锁定对话行
        :return:
        """
        conversation = (
            await ai_conversation_dao.get_by_conversation_id_for_update(db, conversation_id)
            if for_update
            else await ai_conversation_dao.get_by_conversation_id(db, conversation_id)
        )
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
                row_model_message_ranges=[],
            )
        message_rows = list(await ai_message_dao.get_all_by_message_index(db, conversation_id))
        if require_messages and not message_rows:
            raise errors.RequestError(msg='对话消息不存在')
        model_messages, row_model_message_ranges = expand_message_rows(message_rows)
        return ChatConversationState(
            conversation=conversation,
            message_rows=message_rows,
            model_messages=model_messages,
            row_model_message_ranges=row_model_message_ranges,
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
        message_rows = await ai_message_dao.get_all_by_message_index(db, conversation.conversation_id)
        model_messages, row_model_message_ranges = expand_message_rows(message_rows)
        message_ids, provider_ids, model_ids, message_indexes = expand_message_row_metadata(
            message_rows,
            row_model_message_ranges,
        )
        protocol_adapter = get_chat_protocol_adapter()
        messages_snapshot = protocol_adapter.serialize_messages_to_snapshot(
            model_messages,
            conversation_id=conversation.conversation_id,
            message_ids=message_ids,
            provider_ids=provider_ids,
            model_ids=model_ids,
            message_indexes=message_indexes,
        )
        return GetAIConversationDetail(
            id=conversation.id,
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            is_pinned=conversation.pinned_time is not None,
            provider_id=conversation.provider_id,
            model_id=conversation.model_id,
            created_time=conversation.created_time,
            updated_time=conversation.updated_time,
            is_generating=await has_active_run(conversation.conversation_id),
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
                'is_generating': await has_active_run(item['conversation_id']),
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
            for_update=True,
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
            for_update=True,
        )
        return await ai_conversation_dao.update_pinned_time(
            db,
            conversation.id,
            timezone.now() if obj.is_pinned else None,
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
            for_update=True,
        )
        await self.ensure_idle(db=db, conversation_id=conversation_id)
        await ai_message_dao.delete(db, conversation_id)
        return await ai_conversation_dao.delete(db, conversation_id, user_id)


ai_conversation_service: AIConversationService = AIConversationService()
