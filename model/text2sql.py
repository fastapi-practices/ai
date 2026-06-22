import sqlalchemy as sa

from sqlalchemy.orm import Mapped, mapped_column

from backend.common.model import Base, UniversalText, id_key


class AIText2SqlDataset(Base):
    """AI Text2SQL 数据集（表与样例的容器，chat 按数据集圈定可见范围）"""

    __tablename__ = 'ai_text2sql_dataset'

    id: Mapped[id_key] = mapped_column(init=False)
    name: Mapped[str] = mapped_column(sa.String(128), comment='数据集名称')
    description: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='描述')
    enabled: Mapped[int] = mapped_column(default=1, comment='是否启用（0停用 1启用）')
    sort: Mapped[int] = mapped_column(default=0, comment='排序')


class AIText2SqlTable(Base):
    """AI Text2SQL 已选数据表（数据源管理）"""

    __tablename__ = 'ai_text2sql_table'

    id: Mapped[id_key] = mapped_column(init=False)
    dataset_id: Mapped[int] = mapped_column(sa.BigInteger, index=True, comment='所属数据集 ID')
    table_name: Mapped[str] = mapped_column(sa.String(128), comment='表名')
    schema_name: Mapped[str] = mapped_column(sa.String(64), default='fba', comment='库名/schema')
    table_comment: Mapped[str | None] = mapped_column(sa.String(256), default=None, comment='表注释')
    custom_desc: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='自定义语义描述')
    enabled: Mapped[int] = mapped_column(default=1, comment='是否启用（0停用 1启用）')
    sort: Mapped[int] = mapped_column(default=0, comment='排序')


class AIText2SqlExample(Base):
    """AI Text2SQL Few-shot 样例"""

    __tablename__ = 'ai_text2sql_example'

    id: Mapped[id_key] = mapped_column(init=False)
    dataset_id: Mapped[int] = mapped_column(sa.BigInteger, index=True, comment='所属数据集 ID')
    question: Mapped[str] = mapped_column(UniversalText, comment='自然语言问题')
    sql: Mapped[str] = mapped_column(UniversalText, comment='示范 SQL')
    related_tables: Mapped[str | None] = mapped_column(sa.String(512), default=None, comment='相关表（逗号分隔）')
    note: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='备注')
    enabled: Mapped[int] = mapped_column(default=1, comment='是否启用（0停用 1启用）')
    sort: Mapped[int] = mapped_column(default=0, comment='排序')


class AIText2SqlHistory(Base):
    """AI Text2SQL 查询历史"""

    __tablename__ = 'ai_text2sql_history'

    id: Mapped[id_key] = mapped_column(init=False)
    user_id: Mapped[int] = mapped_column(sa.BigInteger, comment='用户 ID')
    question: Mapped[str] = mapped_column(UniversalText, comment='自然语言问题')
    sql: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='生成 SQL')
    executed: Mapped[int] = mapped_column(default=0, comment='是否已执行（0否 1是）')
    row_count: Mapped[int] = mapped_column(default=0, comment='结果行数')
    error: Mapped[str | None] = mapped_column(UniversalText, default=None, comment='错误信息')
    duration_ms: Mapped[int] = mapped_column(default=0, comment='耗时（毫秒）')
