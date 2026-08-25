import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, id_key


class AIDefaultModel(Base):
    """AI 默认模型"""

    __tablename__ = 'ai_default_model'
    __table_args__ = (
        sa.UniqueConstraint('scene', 'deleted', name='uk_ai_default_model_scene_deleted'),
        {'comment': 'AI 默认模型'},
    )

    id: Mapped[id_key] = mapped_column(init=False)
    scene: Mapped[str] = mapped_column(sa.String(32), comment='默认模型场景')
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    status: Mapped[int] = mapped_column(default=1, comment='状态（0停用 1正常）')
