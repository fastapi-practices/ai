from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import AIText2SqlExample
from backend.plugin.ai.schema.text2sql import CreateText2SqlExampleParam, UpdateText2SqlExampleParam
from backend.utils.timezone import timezone


class CRUDText2SqlExample(CRUDPlus[AIText2SqlExample]):
    """Text2SQL Few-shot 样例数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> AIText2SqlExample | None:
        """
        获取样例

        :param db: 数据库会话
        :param pk: 记录 ID
        :return:
        """
        return await self.select_model(db, pk, deleted=0)

    async def get_all_enabled(self, db: AsyncSession, dataset_id: int | None = None) -> Sequence[AIText2SqlExample]:
        """
        获取全部启用的样例（供 Agent 召回）

        :param db: 数据库会话
        :param dataset_id: 数据集 ID；传入则仅返回该数据集的样例，不传则返回全部
        :return:
        """
        filters: dict = {'enabled': 1, 'deleted': 0}
        if dataset_id is not None:
            filters['dataset_id'] = dataset_id
        return await self.select_models(db, **filters)

    async def get_select(
        self,
        question: str | None,
        enabled: int | None,
        dataset_id: int | None = None,
    ) -> Select:
        """
        获取样例分页查询

        :param question: 自然语言问题（模糊）
        :param enabled: 是否启用
        :param dataset_id: 所属数据集 ID
        :return:
        """
        filters = {'deleted': 0}
        if dataset_id is not None:
            filters.update(dataset_id=dataset_id)
        if question is not None:
            filters.update(question__like=f'%{question}%')
        if enabled is not None:
            filters.update(enabled=enabled)
        return await self.select_order('sort', 'asc', **filters)

    async def create(self, db: AsyncSession, obj: CreateText2SqlExampleParam) -> None:
        """
        创建样例

        :param db: 数据库会话
        :param obj: 创建参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateText2SqlExampleParam) -> int:
        """
        更新样例

        :param db: 数据库会话
        :param pk: 记录 ID
        :param obj: 更新参数
        :return:
        """
        return await self.update_model_by_column(db, obj, id=pk, deleted=0)

    async def delete(self, db: AsyncSession, pk: int) -> int:
        """
        删除样例（逻辑删除）

        :param db: 数据库会话
        :param pk: 记录 ID
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


text2sql_example_dao: CRUDText2SqlExample = CRUDText2SqlExample(AIText2SqlExample)
