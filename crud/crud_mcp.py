from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_crud_plus import CRUDPlus

from backend.plugin.ai.model import Mcp
from backend.plugin.ai.schema.mcp import CreateMcpParam, UpdateMcpParam


class CRUDMcp(CRUDPlus[Mcp]):
    """MCP 数据库操作类"""

    async def get(self, db: AsyncSession, pk: int) -> Mcp | None:
        """
        获取 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :return:
        """
        return await self.select_model(db, pk)

    async def get_by_name(self, db: AsyncSession, name: str) -> Mcp | None:
        """
        通过名称获取 MCP

        :param db: 数据库会话
        :param name: MCP 名称
        :return:
        """
        return await self.select_model_by_column(db, name=name)

    async def get_select(self, name: str | None, type: int | None) -> Select:
        """
        获取 MCP 列表

        :param name: MCP 名称
        :param type: MCP 类型
        :return:
        """
        filters = {}

        if name is not None:
            filters.update(name__like=f'%{name}%')
        if type is not None:
            filters.update(type=type)

        return await self.select_order('id', 'desc', **filters)

    async def get_all(self, db: AsyncSession) -> Sequence[Mcp]:
        """
        获取所有 MCP

        :param db: 数据库会话
        :return:
        """
        return await self.select_models(db)

    async def create(self, db: AsyncSession, obj: CreateMcpParam) -> None:
        """
        创建 MCP

        :param db: 数据库会话
        :param obj: 创建 MCP 参数
        :return:
        """
        await self.create_model(db, obj)

    async def update(self, db: AsyncSession, pk: int, obj: UpdateMcpParam) -> int:
        """
        更新 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :param obj: 更新 MCP 参数
        :return:
        """
        return await self.update_model(db, pk, obj)

    async def delete(self, db: AsyncSession, pk: int) -> int:
        """
        删除 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :return:
        """
        return await self.delete_model(db, pk)


mcp_dao: CRUDMcp = CRUDMcp(Mcp)
