"""Text2SQL 安全护栏。

设计原则（fail-closed）：
1. 只允许**单条只读 SELECT**（基于 sqlparse.get_type，权威判定，避免关键字误伤列名）。
2. 拒绝多语句（防 `;` 注入第二条写语句）。
3. 拒绝 `SELECT ... INTO`（PG 建表 / MySQL 写文件·变量等副作用）。
4. **表白名单**：用正则全文扫描 FROM/JOIN 引用表，覆盖子查询体、CTE 体、UNION 等
   任意嵌套位置——管理员控制 AI 可见表的边界，防越表取数（如 sys_user 密码）。
   CTE 名视为查询内合法“虚拟表”，不要求在白名单中。
5. 缺 LIMIT 时注入 `LIMIT <max_rows>`；已有 LIMIT 则保留（行数上限在执行层兜底）。

本模块保持纯净（仅依赖 sqlparse + re + 自身 exceptions），便于无 DB 单测。
真正的只读权限与超时由执行层（只读账号 + readonly_db）兜底，护栏是应用层第一道关。

已知 v1 局限（由只读账号兜底，且为 fail-closed 偏向拒绝）：
- 注释/字符串字面量里出现 `from/join <词>` 可能被误当作表名而拒绝（极少见）。
- 列名恰好为 `from`（如 `SELECT from FROM t`）等保留字边界场景。
"""

import re

import sqlparse

from backend.plugin.ai.text2sql.exceptions import TableNotAllowedError, UnsafeSqlError

_INTO_RE = re.compile(r'\binto\b', re.IGNORECASE)
_HAS_LIMIT_RE = re.compile(r'\blimit\b', re.IGNORECASE)
# FROM/JOIN 后的表名（支持 schema.table，取末段去引号）
_TABLE_REF_RE = re.compile(r'\b(?:from|join)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)', re.IGNORECASE)
# CTE 名称：WITH [RECURSIVE] name AS ( 或 , name AS (
_CTE_NAME_RE = re.compile(r'(?:\bwith\b\s+(?:recursive\s+)?|,)\s*([A-Za-z_]\w*)\s+as\s*\(', re.IGNORECASE)


def _extract_table_names(text: str) -> set[str]:
    """
    全文提取 FROM/JOIN 引用的表名（覆盖子查询体 / CTE 体 / UNION 等嵌套位置）

    :param text: SQL 文本
    :return: 小写裸表名集合
    """
    names: set[str] = set()
    for match in _TABLE_REF_RE.findall(text):
        names.add(match.split('.')[-1].strip().lower())
    return names


def _extract_cte_names(text: str) -> set[str]:
    """提取 CTE 名称（视为查询内合法的“虚拟表”，不要求在 DB 白名单中）"""
    return {name.lower() for name in _CTE_NAME_RE.findall(text)}


def validate_and_normalize(sql: str, *, allowlist: set[str], max_rows: int) -> str:
    """
    校验并归一化 SQL：仅放行单条只读 SELECT，强制表白名单与 LIMIT

    :param sql: 待校验 SQL
    :param allowlist: 允许查询的表名集合（小写裸名）
    :param max_rows: 缺省 LIMIT 行数上限
    :return: 归一化后的安全 SQL（缺 LIMIT 时已注入）
    :raises UnsafeSqlError: 空 / 多语句 / 非 SELECT / 含 INTO 等
    :raises TableNotAllowedError: 引用了不在白名单内的表
    """
    if not sql or not sql.strip():
        raise UnsafeSqlError('SQL 为空')

    statements = [s for s in sqlparse.parse(sql) if s.token_first(skip_ws=True, skip_cm=True) is not None]
    if len(statements) != 1:
        raise UnsafeSqlError('仅允许单条 SQL 语句')

    statement = statements[0]
    if statement.get_type() != 'SELECT':
        raise UnsafeSqlError('仅允许 SELECT 查询')

    text = str(statement)
    if _INTO_RE.search(text):
        raise UnsafeSqlError('禁止 SELECT ... INTO 等带副作用写操作')

    referenced = _extract_table_names(text)
    allowed = {name.lower() for name in allowlist} | _extract_cte_names(text)
    disallowed = referenced - allowed
    if disallowed:
        raise TableNotAllowedError(f'引用了未授权的表: {", ".join(sorted(disallowed))}')

    normalized = statement.value.strip().rstrip(';').strip()
    if not _HAS_LIMIT_RE.search(normalized):
        normalized = f'{normalized}\nLIMIT {max(1, int(max_rows))}'

    return normalized
