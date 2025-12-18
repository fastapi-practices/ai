import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, UniversalText, id_key


class AiProvider(Base):
    """AI 供应商"""

    __tablename__ = 'ai_provider'

    id: Mapped[id_key] = mapped_column(init=False)
    name: Mapped[str] = mapped_column(sa.String(256), comment='供应商名称')
    type: Mapped[int] = mapped_column(comment='供应商类型（0OpenAI 1Anthropic 2Gemini）')
    api_key: Mapped[str] = mapped_column(UniversalText, comment='API Key')
    api_host: Mapped[str] = mapped_column(sa.String(512), comment='API Host')
    status: Mapped[int] = mapped_column(default=1, comment='角色状态（0停用 1正常）')
    remark: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='备注')
