import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, id_key


class AIProvider(Base):
    """AI 供应商"""

    __tablename__ = 'ai_provider'

    id: Mapped[id_key] = mapped_column(init=False)
    name: Mapped[str] = mapped_column(sa.String(256), comment='供应商名称')
    type: Mapped[int] = mapped_column(comment='供应商类型')
    api_key: Mapped[str] = mapped_column(sa.Text, comment='接口密钥')
    api_host: Mapped[str] = mapped_column(sa.String(512), comment='接口地址')
    status: Mapped[int] = mapped_column(default=1, comment='供应商状态（0停用 1正常）')
    remark: Mapped[str | None] = mapped_column(sa.Text, default=None, comment='备注')
