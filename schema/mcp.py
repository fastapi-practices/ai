from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, HttpUrl

from backend.common.schema import SchemaBase
from backend.plugin.ai.enums import McpType


class McpSchemaBase(SchemaBase):
    """MCP 基础模型"""

    name: str = Field(description='MCP 名称')
    command: str = Field(description='MCP 命令')
    type: McpType = Field(McpType.stdio, description='MCP 类型')
    description: str | None = Field(None, description='MCP 描述')
    url: HttpUrl | None = Field(None, description='MCP 端点链接')
    headers: dict[str, Any] | None = Field(None, description='请求 MCP 端点时的请求头')
    args: list[str] | None = Field(None, description='MCP 命令参数')
    env: dict[str, Any] | None = Field(None, description='MCP 环境变量')
    timeout: float = Field(5, description='客户端初始化超时时间（秒）')
    read_timeout: float | None = Field(5 * 60, description='等待新消息的最长时间（秒）')


class CreateMcpParam(McpSchemaBase):
    """创建 MCP 参数"""


class UpdateMcpParam(McpSchemaBase):
    """更新 MCP 参数"""


class GetMcpDetail(McpSchemaBase):
    """MCP 详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='MCP ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')
