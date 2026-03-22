import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, UniversalText, id_key


class Mcp(Base):
    """MCP 表"""

    __tablename__ = 'ai_mcp'

    id: Mapped[id_key] = mapped_column(init=False)
    name: Mapped[str] = mapped_column(sa.String(64), unique=True, comment='MCP 名称')
    type: Mapped[int] = mapped_column(default=0, comment='MCP 类型（0stdio 1sse 2streamable_http）')
    description: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='MCP 描述')
    url: Mapped[str | None] = mapped_column(sa.String(256), default=None, comment='MCP 端点链接')
    headers: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='请求 MCP 端点时的请求头')
    command: Mapped[str | None] = mapped_column(sa.String(256), default=None, comment='MCP 命令')
    args: Mapped[str | None] = mapped_column(sa.JSON(), default=None, comment='MCP 命令参数')
    env: Mapped[str | None] = mapped_column(sa.JSON(), default=None, comment='MCP 环境变量')
    timeout: Mapped[float | None] = mapped_column(default=5, comment='客户端初始化超时时间（秒）')
    read_timeout: Mapped[float | None] = mapped_column(default=5 * 60, comment='等待新消息的最长时间（秒）')
