from collections.abc import Sequence

from pydantic_ai.capabilities import Toolset
from pydantic_ai.mcp import MCPToolset, SSETransport, StdioTransport, StreamableHttpTransport

from backend.common.exception import errors
from backend.core.conf import settings
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import McpType
from backend.plugin.ai.service.mcp_service import mcp_service


async def build_mcp_capability(ctx: CapabilityContext) -> Sequence[CapabilityResult]:
    """
    构建 MCP 能力

    :param ctx: 能力构建上下文
    :return:
    """
    if not ctx.forwarded_props.mcp_ids:
        return ()
    mcps = await mcp_service.get_by_ids(db=ctx.db, mcp_ids=ctx.forwarded_props.mcp_ids)
    results: list[CapabilityResult] = []
    for mcp in mcps:
        headers = {str(key): str(value) for key, value in (mcp.headers or {}).items()}
        tool_prefix = (mcp.tool_prefix or f'mcp_{mcp.id}').rstrip('_') or f'mcp_{mcp.id}'

        if mcp.type == McpType.stdio:
            transport = StdioTransport(
                command=mcp.command,
                args=mcp.args or [],
                env={str(key): str(value) for key, value in (mcp.env or {}).items()},
            )
        elif mcp.type == McpType.sse:
            if not mcp.url:
                raise errors.RequestError(msg=f'MCP 缺少 SSE URL: {mcp.name}')
            transport = SSETransport(
                url=mcp.url,
                headers=headers,
                sse_read_timeout=mcp.read_timeout,
            )
        else:
            if not mcp.url:
                raise errors.RequestError(msg=f'MCP 缺少 Streamable HTTP URL: {mcp.name}')
            transport = StreamableHttpTransport(
                url=mcp.url,
                headers=headers,
                sse_read_timeout=mcp.read_timeout,
            )

        toolset = MCPToolset(
            transport,
            id=f'mcp_{mcp.id}',
            max_retries=settings.AI_MCP_MAX_RETRIES,
            include_instructions=mcp.include_instructions,
            init_timeout=mcp.timeout,
            read_timeout=mcp.read_timeout,
        )
        results.append(
            CapabilityResult(
                capability=Toolset(toolset.prefixed(tool_prefix)),
                introduces_function_tool_source=True,
            )
        )
    return tuple(results)
