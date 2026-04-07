import json

from collections.abc import Sequence
from typing import Any

from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.pagination import paging_data
from backend.plugin.ai.crud.crud_mcp import mcp_dao
from backend.plugin.ai.enums import McpType
from backend.plugin.ai.model import Mcp
from backend.plugin.ai.schema.mcp import CreateMcpParam, UpdateMcpParam

McpToolset = MCPServerStdio | MCPServerSSE | MCPServerStreamableHTTP


class McpService:
    """MCP 服务类"""

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
    async def get_all(*, db: AsyncSession) -> Sequence[Mcp]:
        """
        获取所有 MCP

        :param db: 数据库会话
        :return:
        """
        return await mcp_dao.get_all(db)

    @staticmethod
    async def get_toolsets(*, db: AsyncSession, mcp_ids: list[int]) -> list[McpToolset]:
        """
        获取 MCP 工具集

        :param db: 数据库会话
        :param mcp_ids: MCP ID 列表
        :return:
        """
        mcps = await mcp_dao.get_by_ids(db, mcp_ids)
        toolsets: list[McpToolset] = []
        for mcp in mcps:
            headers = json.loads(mcp.headers) if isinstance(mcp.headers, str) else mcp.headers
            if headers is not None and not isinstance(headers, dict):
                raise errors.RequestError(msg=f'MCP 请求头格式非法: {mcp.name}')
            parsed_headers = None if headers is None else {str(key): str(value) for key, value in headers.items()}
            if mcp.type == McpType.stdio:
                args = json.loads(mcp.args) if isinstance(mcp.args, str) else (mcp.args or [])
                env = json.loads(mcp.env) if isinstance(mcp.env, str) else (mcp.env or {})
                if not isinstance(args, list):
                    raise errors.RequestError(msg=f'MCP 命令参数格式非法: {mcp.name}')
                if not isinstance(env, dict):
                    raise errors.RequestError(msg=f'MCP 环境变量格式非法: {mcp.name}')
                toolsets.append(
                    MCPServerStdio(
                        command=mcp.command,
                        args=[str(arg) for arg in args],
                        env={str(key): str(value) for key, value in env.items()},
                        timeout=mcp.timeout,
                    )
                )
            elif mcp.type == McpType.sse:
                if not mcp.url:
                    raise errors.RequestError(msg=f'MCP 缺少 SSE URL: {mcp.name}')
                toolsets.append(
                    MCPServerSSE(
                        url=mcp.url,
                        headers=parsed_headers,
                        timeout=mcp.timeout,
                        read_timeout=mcp.read_timeout,
                    )
                )
            else:
                if not mcp.url:
                    raise errors.RequestError(msg=f'MCP 缺少 Streamable HTTP URL: {mcp.name}')
                toolsets.append(
                    MCPServerStreamableHTTP(
                        url=mcp.url,
                        headers=parsed_headers,
                        timeout=mcp.timeout,
                        read_timeout=mcp.read_timeout,
                    )
                )
        return toolsets

    @staticmethod
    async def get_list(*, db: AsyncSession, name: str | None, type: int | None) -> dict[str, Any]:
        """
        获取 MCP 列表

        :param db: 数据库会话
        :param name: MCP 名称
        :param type: MCP 类型
        :return:
        """
        mcp_select = await mcp_dao.get_select(name=name, type=type)
        return await paging_data(db, mcp_select)

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
        return await mcp_dao.update(db, pk, obj)

    @staticmethod
    async def delete(*, db: AsyncSession, pk: int) -> int:
        """
        删除 MCP

        :param db: 数据库会话
        :param pk: MCP ID
        :return:
        """
        return await mcp_dao.delete(db, pk)


mcp_service: McpService = McpService()
