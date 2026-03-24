from datetime import datetime

from pydantic import ConfigDict, Field

from backend.common.schema import SchemaBase


class AIQuickPhraseSchemaBase(SchemaBase):
    """AI 快捷短语基础模型"""

    title: str = Field(description='短语标题')
    content: str = Field(description='短语内容')
    sort: int = Field(default=0, description='排序')


class CreateAIQuickPhraseParam(AIQuickPhraseSchemaBase):
    """创建 AI 快捷短语参数"""


class UpdateAIQuickPhraseParam(AIQuickPhraseSchemaBase):
    """更新 AI 快捷短语参数"""


class GetAIQuickPhraseDetail(AIQuickPhraseSchemaBase):
    """AI 快捷短语详情"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description='快捷短语 ID')
    user_id: int = Field(description='用户 ID')
    created_time: datetime = Field(description='创建时间')
    updated_time: datetime | None = Field(default=None, description='更新时间')
