import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, UniversalText, id_key


class AIModel(Base):
    """AI 模型"""

    __tablename__ = 'ai_model'
    __table_args__ = (
        sa.UniqueConstraint('provider_id', 'model_id', 'deleted', name='uk_ai_model_provider_id_model_id_deleted'),
        {'comment': 'AI 模型'},
    )

    id: Mapped[id_key] = mapped_column(init=False)
    provider_id: Mapped[int] = mapped_column(sa.BigInteger, comment='供应商关联 ID')
    model_id: Mapped[str] = mapped_column(sa.String(512), comment='模型 ID')
    status: Mapped[int] = mapped_column(default=1, comment='模型状态（0停用 1正常）')
    context_max_part_chars: Mapped[int | None] = mapped_column(
        default=None,
        comment='单个模型响应部分允许保留的最大字符数',
    )
    context_max_messages: Mapped[int | None] = mapped_column(
        default=None,
        comment='发送模型前触发历史消息裁剪的消息数量',
    )
    context_keep_messages: Mapped[int] = mapped_column(
        default=60,
        server_default='60',
        comment='裁剪后保留的最近消息数量',
    )
    context_max_tokens: Mapped[int | None] = mapped_column(
        default=None,
        comment='上下文容量告警使用的最大 token 数量（空值关闭）',
    )
    remark: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='备注')
