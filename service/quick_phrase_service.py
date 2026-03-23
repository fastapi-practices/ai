from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_quick_phrase import ai_quick_phrase_dao
from backend.plugin.ai.model import AIQuickPhrase
from backend.plugin.ai.schema.quick_phrase import (
    CreateAIQuickPhraseParam,
    UpdateAIQuickPhraseParam,
)


class AIQuickPhraseService:
    """AI 快捷短语服务"""

    @staticmethod
    async def get(*, db: AsyncSession, pk: int, user_id: int) -> AIQuickPhrase:
        """
        获取快捷短语详情

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param user_id: 用户 ID
        :return:
        """
        quick_phrase = await ai_quick_phrase_dao.get_by_id_and_user_id(db, pk, user_id)
        if not quick_phrase:
            raise errors.NotFoundError(msg='快捷短语不存在')
        return quick_phrase

    @staticmethod
    async def get_all(*, db: AsyncSession, user_id: int) -> Sequence[AIQuickPhrase]:
        """
        获取当前用户所有快捷短语

        :param db: 数据库会话
        :param user_id: 用户 ID
        :return:
        """
        return await ai_quick_phrase_dao.get_all_by_user_id(db, user_id)

    @staticmethod
    async def get_list(*, db: AsyncSession, user_id: int, content: str | None) -> dict[str, Any]:
        """
        获取当前用户快捷短语列表

        :param db: 数据库会话
        :param user_id: 用户 ID
        :param content: 短语内容
        :return:
        """
        quick_phrase_select = await ai_quick_phrase_dao.get_select(user_id, content)
        return await paging_data(db, quick_phrase_select)

    @staticmethod
    async def create(*, db: AsyncSession, obj: CreateAIQuickPhraseParam, user_id: int) -> None:
        """
        创建快捷短语

        :param db: 数据库会话
        :param obj: 创建参数
        :param user_id: 用户 ID
        :return:
        """
        content = obj.content.strip()
        if not content:
            raise errors.RequestError(msg='快捷短语内容不能为空')
        if len(content) > 100:
            raise errors.RequestError(msg='快捷短语内容过长')
        await ai_quick_phrase_dao.create(db, obj.model_copy(update={'content': content}), user_id)

    async def update(self, *, db: AsyncSession, pk: int, obj: UpdateAIQuickPhraseParam, user_id: int) -> int:
        """
        更新快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param obj: 更新参数
        :param user_id: 用户 ID
        :return:
        """
        await self.get(db=db, pk=pk, user_id=user_id)
        content = obj.content.strip()
        if not content:
            raise errors.RequestError(msg='快捷短语内容不能为空')
        if len(content) > 100:
            raise errors.RequestError(msg='快捷短语内容过长')
        return await ai_quick_phrase_dao.update(db, pk, obj.model_copy(update={'content': content}))

    async def delete(self, *, db: AsyncSession, pk: int, user_id: int) -> int:
        """
        删除快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param user_id: 用户 ID
        :return:
        """
        await self.get(db=db, pk=pk, user_id=user_id)
        return await ai_quick_phrase_dao.delete(db, pk, user_id)


ai_quick_phrase_service: AIQuickPhraseService = AIQuickPhraseService()
