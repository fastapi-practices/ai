from pydantic import Field

from backend.common.schema import SchemaBase


class UpdateAIMessageParam(SchemaBase):
    """更新消息参数"""

    content: str = Field(description='消息内容')
