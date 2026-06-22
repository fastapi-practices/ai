"""Text2SQL 异常定义。


服务层负责将这些异常映射为对应的 HTTP 错误（见 backend/common/exception/errors）：
- UnsafeSqlError        -> 400（SQL 未通过安全护栏）
- TableNotAllowedError  -> 400（引用了未挑选/未授权的表）
- Text2SqlTimeoutError  -> 408（执行超时）
"""


class Text2SqlError(Exception):
    """Text2SQL 基础异常"""


class UnsafeSqlError(Text2SqlError):
    """SQL 未通过安全护栏（非只读 / 多语句 / 危险关键字 等）"""


class TableNotAllowedError(UnsafeSqlError):
    """SQL 引用了未挑选或未授权的表"""


class Text2SqlTimeoutError(Text2SqlError):
    """SQL 执行超时"""
