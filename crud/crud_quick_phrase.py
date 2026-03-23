from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIQuickPhrase
from backend.plugin.ai.schema.quick_phrase import CreateAIQuickPhraseParam, UpdateAIQuickPhraseParam


class CRUDAIQuickPhrase(CRUDPlus[AIQuickPhrase]):
    """AI 快捷短语数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIQuickPhrase | None:
        """
        获取 AI 快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :return:
        """
        return await self.select_model(db, pk)

    async def get_by_id_and_user_id(self, db: AsyncSession, pk: int, user_id: int) -> AIQuickPhrase | None:
        """
        获取当前用户的快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param user_id: 用户 ID
        :return:
        """
        return await self.select_model_by_column(db, id=pk, user_id=user_id)

    async def get_select(self, user_id: int, content: str | None) -> Select:
        """
        获取快捷短语列表查询表达式

        :param user_id: 用户 ID
        :param content: 短语内容
        :return:
        """
        filters = {'user_id': user_id}

        if content is not None:
            filters['content__like'] = f'%{content}%'

        return await self.select_order('sort', 'asc', **filters)

    async def get_all_by_user_id(self, db: AsyncSession, user_id: int) -> Sequence[AIQuickPhrase]:
        """
        获取当前用户的所有快捷短语

        :param db: 数据库会话
        :param user_id: 用户 ID
        :return:
        """
        return await self.select_models_order(db, 'sort', 'asc', user_id=user_id)

    async def create(self, db: AsyncSession, obj: CreateAIQuickPhraseParam, user_id: int) -> None:
        """
        创建 AI 快捷短语

        :param db: 数据库会话
        :param obj: 创建参数
        :param user_id: 用户 ID
        :return:
        """
        quick_phrase = self.model(**obj.model_dump(), user_id=user_id)
        db.add(quick_phrase)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateAIQuickPhraseParam) -> int:
        """
        更新 AI 快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param obj: 更新参数
        :return:
        """
        return await self.update_model(db, pk, obj)

    async def delete(self, db: AsyncSession, pk: int, user_id: int) -> int:
        """
        删除 AI 快捷短语

        :param db: 数据库会话
        :param pk: 快捷短语 ID
        :param user_id: 用户 ID
        :return:
        """
        return await self.delete_model_by_column(db, id=pk, user_id=user_id)


ai_quick_phrase_dao: CRUDAIQuickPhrase = CRUDAIQuickPhrase(AIQuickPhrase)
