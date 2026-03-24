from typing import Any

import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, id_key


class AIChatMessage(Base):
    """AI 聊天消息"""

    __tablename__ = 'ai_chat_message'

    id: Mapped[id_key] = mapped_column(init=False)
    conversation_id: Mapped[str] = mapped_column(sa.String(64), index=True, comment='会话 ID')
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    message_index: Mapped[int] = mapped_column(index=True, comment='消息索引')
    message: Mapped[dict[str, Any]] = mapped_column(sa.JSON(), comment='原始模型消息')
