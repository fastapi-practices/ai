from collections.abc import Sequence

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.plugin.ai.crud.crud_mcp import mcp_dao
from backend.plugin.ai.model import Mcp
from backend.plugin.ai.schema.mcp import CreateMcpParam, UpdateMcpParam


class McpService:
    @staticmethod
    async def get(*, db: AsyncSession, pk: int) -> Mcp:
        """
        获取 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :return:
        """
        mcp = await mcp_dao.get(db, pk)
        if not mcp:
            raise errors.NotFoundError(msg='MCP 不存在')
        return mcp

    @staticmethod
    async def get_select(*, name: str | None, type: int | None) -> Select:
        """
        获取 MCP 查询对象

        :param name: MCP 名称
        :param type: MCP 类型
        :return:
        """
        return await mcp_dao.get_select(name=name, type=type)

    @staticmethod
    async def get_all(*, db: AsyncSession) -> Sequence[Mcp]:
        """
        获取所有 MCP

        :param db: 数据库会话
        :return:
        """
        mcps = await mcp_dao.get_all(db)
        return mcps

    @staticmethod
    async def create(*, db: AsyncSession, obj: CreateMcpParam) -> None:
        """
        创建 MCP

        :param db: 数据库会话
        :param obj: 创建 MCP 参数
        :return:
        """
        mcp = await mcp_dao.get_by_name(db, name=obj.name)
        if mcp:
            raise errors.ForbiddenError(msg='MCP 已存在')
        await mcp_dao.create(db, obj)

    @staticmethod
    async def update(*, db: AsyncSession, pk: int, obj: UpdateMcpParam) -> int:
        """
        更新 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :param obj: 更新 MCP 参数
        :return:
        """
        mcp = await mcp_dao.get(db, pk)
        if not mcp:
            raise errors.NotFoundError(msg='MCP 不存在')
        if mcp.name != obj.name and await mcp_dao.get_by_name(db, name=obj.name):
            raise errors.ForbiddenError(msg='MCP 已存在')
        count = await mcp_dao.update(db, pk, obj)
        return count

    @staticmethod
    async def delete(*, db: AsyncSession, pk: int) -> int:
        """
        删除 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :return:
        """
        count = await mcp_dao.delete(db, pk)
        return count


mcp_service: McpService = McpService()
