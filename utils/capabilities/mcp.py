from typing import Any

from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.enums import McpType
from backend.plugin.ai.model import Mcp


def build_mcp_capability(*, mcp: Mcp) -> AbstractCapability[Any]:
    """
    构建 MCP 能力

    :param mcp: MCP 配置
    :return:
    """
    headers = {str(key): str(value) for key, value in (mcp.headers or {}).items()}
    tool_prefix = (mcp.tool_prefix or f'mcp_{mcp.id}').rstrip('_') or f'mcp_{mcp.id}'

    if mcp.type == McpType.stdio:
        mcp_server = MCPServerStdio(
            command=mcp.command,
            args=mcp.args or [],
            env={str(key): str(value) for key, value in (mcp.env or {}).items()},
            tool_prefix=tool_prefix,
            timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
            max_retries=settings.AI_MCP_MAX_RETRIES,
            include_instructions=mcp.include_instructions,
        )
    elif mcp.type == McpType.sse:
        if not mcp.url:
            raise errors.RequestError(msg=f'MCP 缺少 SSE URL: {mcp.name}')
        mcp_server = MCPServerSSE(
            url=mcp.url,
            headers=headers,
            tool_prefix=tool_prefix,
            timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
            max_retries=settings.AI_MCP_MAX_RETRIES,
            include_instructions=mcp.include_instructions,
        )
    else:
        if not mcp.url:
            raise errors.RequestError(msg=f'MCP 缺少 Streamable HTTP URL: {mcp.name}')
        mcp_server = MCPServerStreamableHTTP(
            url=mcp.url,
            headers=headers,
            tool_prefix=tool_prefix,
            timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
            max_retries=settings.AI_MCP_MAX_RETRIES,
            include_instructions=mcp.include_instructions,
        )

    return Toolset(mcp_server)
