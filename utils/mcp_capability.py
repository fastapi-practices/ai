import json

from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

from backend.common.exception import errors
from backend.plugin.ai.dataclasses import ChatAgentDeps
from backend.plugin.ai.enums import McpType
from backend.plugin.ai.model import Mcp


def build_mcp_capability(*, mcp: Mcp) -> AbstractCapability[ChatAgentDeps]:
    """
    构建单个 MCP 能力

    :param mcp: MCP 配置
    :return:
    """
    headers = json.loads(mcp.headers) if isinstance(mcp.headers, str) else (mcp.headers or {})
    if not isinstance(headers, dict):
        raise errors.RequestError(msg=f'MCP 请求头格式非法: {mcp.name}')
    parsed_headers = {str(key): str(value) for key, value in headers.items()}

    if mcp.type == McpType.stdio:
        args = json.loads(mcp.args) if isinstance(mcp.args, str) else (mcp.args or [])
        env = json.loads(mcp.env) if isinstance(mcp.env, str) else (mcp.env or {})
        if not isinstance(args, list):
            raise errors.RequestError(msg=f'MCP 命令参数格式非法: {mcp.name}')
        if not isinstance(env, dict):
            raise errors.RequestError(msg=f'MCP 环境变量格式非法: {mcp.name}')
        mcp_server = MCPServerStdio(
            command=mcp.command,
            args=[str(arg) for arg in args],
            env={str(key): str(value) for key, value in env.items()},
            timeout=mcp.timeout,
        )
    elif mcp.type == McpType.sse:
        if not mcp.url:
            raise errors.RequestError(msg=f'MCP 缺少 SSE URL: {mcp.name}')
        mcp_server = MCPServerSSE(
            url=mcp.url,
            headers=parsed_headers,
            timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
        )
    else:
        if not mcp.url:
            raise errors.RequestError(msg=f'MCP 缺少 Streamable HTTP URL: {mcp.name}')
        mcp_server = MCPServerStreamableHTTP(
            url=mcp.url,
            headers=parsed_headers,
            timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
        )

    return Toolset(mcp_server).prefix_tools(f'mcp_{mcp.id}_')
