from datetime import datetime

import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, TimeZone, id_key


class AIConversation(Base):
    """AI 对话"""

    __tablename__ = 'ai_conversation'

    id: Mapped[id_key] = mapped_column(init=False)
    conversation_id: Mapped[str] = mapped_column(sa.String(64), unique=True, index=True, comment='对话 ID')
    user_id: Mapped[int] = mapped_column(sa.BigInteger, index=True, comment='用户 ID')
    title: Mapped[str] = mapped_column(sa.String(256), comment='对话标题')
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    pinned_time: Mapped[datetime | None] = mapped_column(TimeZone, default=None, comment='置顶时间')
    context_start_message_id: Mapped[int | None] = mapped_column(
        sa.BigInteger,
        default=None,
        comment='上下文起始消息 ID',
    )
    context_cleared_time: Mapped[datetime | None] = mapped_column(
        TimeZone,
        default=None,
        comment='上下文清除时间',
    )
