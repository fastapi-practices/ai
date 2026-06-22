from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from backend.common.schema import SchemaBase


# ---------------- 数据集 ----------------


class Text2SqlDatasetSchemaBase(SchemaBase):
    """Text2SQL 数据集基础模型"""

    name: str = Field(description='数据集名称')
    description: str | None = Field(None, description='描述')
    enabled: int = Field(1, description='是否启用（0停用 1启用）')
    sort: int = Field(0, description='排序')


class CreateText2SqlDatasetParam(Text2SqlDatasetSchemaBase):
    """新增数据集"""


class UpdateText2SqlDatasetParam(SchemaBase):
    """更新数据集（部分更新）"""

    name: str | None = Field(None, description='数据集名称')
    description: str | None = Field(None, description='描述')
    enabled: int | None = Field(None, description='是否启用（0停用 1启用）')
    sort: int | None = Field(None, description='排序')


class GetText2SqlDatasetDetail(Text2SqlDatasetSchemaBase):
    """数据集详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class Text2SqlDatasetEnabled(SchemaBase):
    """启用的数据集（chat 选择器用）"""

    id: int = Field(description='数据集 ID')
    name: str = Field(description='数据集名称')
    description: str | None = Field(None, description='描述')


# ---------------- 已选数据表 ----------------


class Text2SqlTableSchemaBase(SchemaBase):
    """Text2SQL 已选数据表基础模型"""

    dataset_id: int = Field(description='所属数据集 ID')
    table_name: str = Field(description='表名')
    schema_name: str = Field('fba', description='库名/schema')
    table_comment: str | None = Field(None, description='表注释')
    custom_desc: str | None = Field(None, description='自定义语义描述（喂给 Agent 提升精度）')
    enabled: int = Field(1, description='是否启用（0停用 1启用）')
    sort: int = Field(0, description='排序')


class CreateText2SqlTableParam(Text2SqlTableSchemaBase):
    """新增已选表"""


class UpdateText2SqlTableParam(SchemaBase):
    """更新已选表（部分更新）"""

    table_comment: str | None = Field(None, description='表注释')
    custom_desc: str | None = Field(None, description='自定义语义描述')
    enabled: int | None = Field(None, description='是否启用（0停用 1启用）')
    sort: int | None = Field(None, description='排序')


class GetText2SqlTableDetail(Text2SqlTableSchemaBase):
    """已选表详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class Text2SqlTableSelectable(SchemaBase):
    """可挑选的数据库表（来自反查，不入库）"""

    table_name: str = Field(description='表名')
    table_comment: str | None = Field(None, description='表注释')
    selected: bool = Field(False, description='是否已挑选')


class Text2SqlExampleSchemaBase(SchemaBase):
    """Text2SQL Few-shot 样例基础模型"""

    dataset_id: int = Field(description='所属数据集 ID')
    question: str = Field(description='自然语言问题')
    sql: str = Field(description='示范 SQL（只读 SELECT）')
    related_tables: str | None = Field(None, description='相关表（逗号分隔，用于召回）')
    note: str | None = Field(None, description='备注')
    enabled: int = Field(1, description='是否启用（0停用 1启用）')
    sort: int = Field(0, description='排序')


class CreateText2SqlExampleParam(Text2SqlExampleSchemaBase):
    """新增 Few-shot 样例"""


class UpdateText2SqlExampleParam(SchemaBase):
    """更新 Few-shot 样例（部分更新）"""

    question: str | None = Field(None, description='自然语言问题')
    sql: str | None = Field(None, description='示范 SQL')
    related_tables: str | None = Field(None, description='相关表（逗号分隔）')
    note: str | None = Field(None, description='备注')
    enabled: int | None = Field(None, description='是否启用（0停用 1启用）')
    sort: int | None = Field(None, description='排序')


class GetText2SqlExampleDetail(Text2SqlExampleSchemaBase):
    """Few-shot 样例详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(None, description='更新时间')


class Text2SqlQueryParam(SchemaBase):
    """自然语言查询参数"""

    question: str = Field(description='自然语言问题')
    dataset_id: int | None = Field(None, description='数据集 ID；不传则使用全部已启用表')


class Text2SqlQueryResult(SchemaBase):
    """自然语言查询结果"""

    sql: str = Field(description='生成的只读 SQL')
    summary: str = Field(description='结果摘要')
    columns: list[str] = Field(default_factory=list, description='结果列')
    rows: list[dict[str, Any]] = Field(default_factory=list, description='结果行（最多 max_rows）')
    row_count: int = Field(0, description='命中总行数')
    duration_ms: int = Field(0, description='耗时（毫秒）')
    history_id: int | None = Field(None, description='历史记录 ID')
