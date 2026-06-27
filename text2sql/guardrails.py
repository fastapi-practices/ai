"""Text2SQL 安全护栏（sqlglot AST，fail-closed）。

相比 v1（sqlparse.get_type + 正则提表名）的关键修正：
1. **表白名单按完整 ``schema.table`` 校验**，不再剥离命名空间——堵死
   ``mysql.user`` / ``information_schema.*`` / ``sys.*`` 因表名碰撞（如业务表恰叫
   ``user``）而越权读取的路径。
2. **强制至少引用一张白名单内的真实表**——阻断无 ``FROM`` 的侦察语句
   （``@@hostname`` / ``USER()`` / ``VERSION()`` / ``SLEEP`` / ``LOAD_FILE`` …）。
3. **拒绝危险函数与系统/会话变量**，即便语句引用了白名单表（如 ``SELECT USER() FROM t``）。
4. 解析失败 / 多语句 / 非 SELECT 一律拒绝（fail-closed）；``DATABASE()`` / ``SCHEMA()``
   / ``SELECT ... INTO OUTFILE`` 等 sqlglot 解析不了的形态自动落入选拒绝分支。
   另对整棵树做写/DDL 节点扫描，防 ``DELETE ... RETURNING`` 嵌入子查询等形态。
5. **LIMIT 服务端钳制到 ≤ max_rows**，防 LLM 生成大 LIMIT 配合 ``.all()`` 造成 OOM。
6. CTE 名视为查询内合法"虚拟表"，不要求落在白名单中。

只读账号（readonly_db）仍是真正的权限边界，本护栏是应用层第一道关。
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from backend.plugin.ai.text2sql.exceptions import TableNotAllowedError, UnsafeSqlError

# 允许的只读根节点类名：SELECT 及集合操作（UNION / INTERSECT / EXCEPT）
_READ_ONLY_ROOT_NAMES = frozenset({'Select', 'Union', 'Intersect', 'Except'})

# 任意位置出现即拒绝的写/DDL 节点类名（用类名匹配，兼容不同 sqlglot 版本的节点命名）
_WRITE_NODE_NAMES = frozenset({
    'Insert', 'Update', 'Delete', 'Drop', 'Alter', 'Create',
    'TruncateTable', 'AlterTable', 'Merge', 'Command',
    'Grant', 'Revoke', 'AddConstraint', 'DropConstraint',
})

# 即便引用了白名单表也一律禁止的函数调用（侦察 / DoS / 文件读 / 锁）
# 注：只匹配 exp.Anonymous（函数调用），列名/别名同名不受影响。
_DANGEROUS_FUNCS = frozenset({
    'load_file',
    'sleep',
    'benchmark',
    'get_lock',
    'release_lock',
    'user',
    'current_user',
    'system_user',
    'session_user',
    'database',
    'schema',
    'connection_id',
    'version',
    'current_database',
    'current_setting',
    'pg_sleep',
    'pg_read_file',
})


def _table_ref(table: exp.Table, default_schema: str) -> str:
    """Table 节点 -> 规范化 ``schema.table``（小写）；未显式指定 schema 时用 default_schema。"""
    schema = (table.db or default_schema).strip().lower()
    return f'{schema}.{table.name.strip().lower()}'


def _cte_names(root: exp.Expression) -> set[str]:
    """查询内定义的 CTE 名（小写），视为合法"虚拟表"。"""
    return {cte.alias.strip().lower() for cte in root.find_all(exp.CTE)}


def _find_dangerous(root: exp.Expression) -> str | None:
    """返回命中的危险描述（用于报错），未命中返回 None。"""
    for node in root.walk():
        if isinstance(node, exp.Parameter):
            # @var / @@var —— 系统/会话变量，侦察用，一律拒
            return f'系统/会话变量不被允许: {node.sql()}'
        if isinstance(node, exp.Anonymous):
            name = (node.name or '').strip().lower()
            if name in _DANGEROUS_FUNCS:
                return f'危险函数不被允许: {node.name.upper()}()'
    return None


def _classify_tables(
    root: exp.Expression,
    *,
    allowlist: set[str],
    ctes: set[str],
    default_schema: str,
) -> tuple[list[str], list[str]]:
    """分类所有 Table 引用。

    :return: (授权的真实表引用, 未授权的表引用)；CTE 引用不计入未授权。
    """
    real: list[str] = []
    disallowed: list[str] = []
    seen: set[str] = set()
    for table in root.find_all(exp.Table):
        ref = _table_ref(table, default_schema)
        if ref in seen:
            continue
        seen.add(ref)
        # CTE 引用（虚拟表）：跳过；其定义体内的真实表由另一个 Table 节点捕获
        if table.name.strip().lower() in ctes:
            continue
        if ref in allowlist:
            real.append(ref)
        else:
            disallowed.append(ref)
    return real, disallowed


def _clamp_limit(query: exp.Expression, max_rows: int) -> None:
    """将最外层 LIMIT 钳制到 ≤ max_rows（原地改 AST）；缺失则注入。"""
    cap = max(1, int(max_rows))
    limit = query.args.get('limit')
    if limit is None:
        query.set('limit', exp.Limit(expression=exp.Literal.number(cap)))
        return
    expr = limit.expression  # type: ignore[attr-defined]
    if isinstance(expr, exp.Literal) and expr.is_int:
        if int(expr.name) > cap:
            limit.set('expression', exp.Literal.number(cap))  # type: ignore[attr-defined]
        return
    # 非字面量（表达式/占位符等）：保守替换为 cap，不冒险评估
    limit.set('expression', exp.Literal.number(cap))  # type: ignore[attr-defined]


def validate_and_normalize(
    sql: str,
    *,
    allowlist: set[str],
    max_rows: int,
    default_schema: str,
    dialect: str | None = None,
) -> str:
    """校验并归一化 SQL：仅放行单条只读 SELECT，强制表白名单、禁侦察、LIMIT 钳制。

    :param sql: 待校验 SQL
    :param allowlist: 允许查询的 ``schema.table`` 集合（小写）
    :param max_rows: 缺省/上限 LIMIT 行数
    :param default_schema: SQL 未显式指定 schema 时使用的默认 schema（即业务库）
    :param dialect: sqlglot 方言（``mysql`` / ``postgres`` 等），默认通用
    :return: 归一化后的安全 SQL（已注入或钳制 LIMIT）
    :raises UnsafeSqlError: 空 / 解析失败 / 多语句 / 非只读 / 含写节点 / 含危险函数或变量 / 无白名单表
    :raises TableNotAllowedError: 引用了不在白名单内的表
    """
    if not sql or not sql.strip():
        raise UnsafeSqlError('SQL 为空')

    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 — 解析失败一律 fail-closed
        raise UnsafeSqlError(f'SQL 解析失败: {exc}') from exc

    statements = [s for s in parsed if s is not None]
    if len(statements) != 1:
        raise UnsafeSqlError('仅允许单条 SQL 语句')

    stmt = statements[0]
    if type(stmt).__name__ not in _READ_ONLY_ROOT_NAMES:
        raise UnsafeSqlError('仅允许只读 SELECT（含 UNION/INTERSECT/EXCEPT）查询')
    for node in stmt.walk():
        if type(node).__name__ in _WRITE_NODE_NAMES:
            raise UnsafeSqlError(f'仅允许只读查询: 检测到 {type(node).__name__}')

    danger = _find_dangerous(stmt)
    if danger:
        raise UnsafeSqlError(danger)

    allowlist = {name.strip().lower() for name in allowlist}
    ctes = _cte_names(stmt)
    real_refs, disallowed = _classify_tables(
        stmt,
        allowlist=allowlist,
        ctes=ctes,
        default_schema=default_schema,
    )
    if disallowed:
        raise TableNotAllowedError(f'引用了未授权的表: {", ".join(sorted(disallowed))}')
    if not real_refs:
        raise UnsafeSqlError('查询必须引用至少一张已授权的数据表（禁止无表侦察语句）')

    _clamp_limit(stmt, max_rows)
    return stmt.sql(dialect=dialect)
