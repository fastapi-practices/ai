from datetime import datetime

import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, TimeZone, id_key


class AIConversation(Base):
    """AI 对话"""

    __tablename__ = 'ai_conversation'
    __table_args__ = (
        sa.UniqueConstraint('conversation_id', 'deleted', name='uk_ai_conversation_conversation_id_deleted'),
        {'comment': 'AI 对话'},
    )

    id: Mapped[id_key] = mapped_column(init=False)
    conversation_id: Mapped[str] = mapped_column(sa.String(64), index=True, comment='对话 ID')
    user_id: Mapped[int] = mapped_column(sa.BigInteger, index=True, comment='用户 ID')
    title: Mapped[str] = mapped_column(sa.String(256), comment='对话标题')
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    pinned_time: Mapped[datetime | None] = mapped_column(TimeZone, default=None, comment='置顶时间')
