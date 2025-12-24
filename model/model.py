import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, UniversalText, id_key


class AIModel(Base):
    """AI 模型"""

    __tablename__ = 'ai_model'

    id: Mapped[id_key] = mapped_column(init=False)
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商关联 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    owned_by: Mapped[str] = mapped_column(sa.String(512), comment='拥有该模型的组织')
    status: Mapped[int] = mapped_column(default=1, comment='角色状态（0停用 1正常）')
    remark: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='备注')
