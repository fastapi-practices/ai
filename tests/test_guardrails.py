"""Text2SQL 安全护栏单测（纯逻辑，无 DB）。

覆盖：放行普通 SELECT/JOIN/CTE/子查询/UNION/已有 LIMIT；拦截非 SELECT、多语句、
INTO OUTFILE、越表（含命名空间碰撞 mysql.user→user、CTE 体/UNION 内越表）、
无表侦察（@@hostname/USER()/SLEEP/LOAD_FILE 等）、大 LIMIT 钳制。
"""

import pytest

from backend.plugin.ai.text2sql.exceptions import TableNotAllowedError, UnsafeSqlError
from backend.plugin.ai.text2sql.guardrails import validate_and_normalize

# 白名单按完整 schema.table 给出（小写）；default_schema 用于 SQL 未显式指定 schema 时
ALLOWLIST = {'public.users', 'public.orders', 'public.products', 'public.user'}
DEFAULT_SCHEMA = 'public'


def _guard(sql: str, **kwargs: object) -> str:
    allowlist = kwargs.get('allowlist', ALLOWLIST)  # type: ignore[arg-type]
    max_rows = kwargs.get('max_rows', 100)  # type: ignore[arg-type]
    default_schema = kwargs.get('default_schema', DEFAULT_SCHEMA)  # type: ignore[arg-type]
    return validate_and_normalize(
        sql,
        allowlist=allowlist,
        max_rows=max_rows,
        default_schema=default_schema,
    )


# ---------------- 放行 ----------------


def test_allows_plain_select_injects_limit() -> None:
    sql = _guard('SELECT * FROM users')
    assert 'LIMIT 100' in sql.upper()


def test_allows_join_and_where() -> None:
    sql = _guard('SELECT u.id FROM users u JOIN orders o ON o.user_id = u.id WHERE u.id > 0')
    assert 'LIMIT 100' in sql.upper()


def test_allows_left_join() -> None:
    sql = _guard('SELECT u.id FROM users u LEFT JOIN orders o ON o.user_id = u.id')
    assert 'LIMIT 100' in sql.upper()


def test_allows_cte() -> None:
    sql = _guard('WITH active AS (SELECT id FROM users WHERE deleted = 0) SELECT * FROM active')
    assert 'LIMIT 100' in sql.upper()


def test_allows_subquery_in_from() -> None:
    sql = _guard('SELECT * FROM (SELECT id FROM users) sub')
    assert 'LIMIT 100' in sql.upper()


def test_allows_union_of_allowlisted_tables() -> None:
    sql = _guard('SELECT id FROM users UNION SELECT id FROM orders')
    assert 'UNION' in sql.upper()
    assert 'LIMIT 100' in sql.upper()


def test_preserves_small_existing_limit() -> None:
    sql = _guard('SELECT * FROM users LIMIT 5')
    assert sql.upper().count('LIMIT') == 1
    assert 'LIMIT 5' in sql.upper().replace(' ', ' ')


def test_accepts_lowercase_select() -> None:
    sql = _guard('select id from users')
    assert 'LIMIT 100' in sql.upper()


def test_accepts_multiline_and_comment() -> None:
    sql = _guard('/* get users */\nSELECT\n  id\nFROM\n  users\nWHERE 1 = 1')
    assert 'LIMIT 100' in sql.upper()


def test_uses_custom_max_rows() -> None:
    sql = _guard('SELECT * FROM users', max_rows=42)
    assert 'LIMIT 42' in sql.upper()


# ---------------- 拦截：非只读 / 多语句 ----------------


@pytest.mark.parametrize(
    ['sql'],
    [
        ["INSERT INTO users (id) VALUES (1)"],
        ["UPDATE users SET name = 'x'"],
        ['DELETE FROM users'],
        ['DROP TABLE users'],
        ['ALTER TABLE users ADD COLUMN x INT'],
        ['TRUNCATE TABLE users'],
        ['CREATE TABLE x (id INT)'],
        ['GRANT SELECT ON users TO bob'],
    ],
)
def test_blocks_non_select(sql: str) -> None:
    with pytest.raises(UnsafeSqlError):
        _guard(sql)


def test_blocks_multi_statement() -> None:
    with pytest.raises(UnsafeSqlError):
        _guard('SELECT * FROM users; DROP TABLE users')


def test_blocks_select_into_outfile() -> None:
    # sqlglot 不解析 INTO OUTFILE → fail-closed
    with pytest.raises(UnsafeSqlError):
        _guard("SELECT * FROM users INTO OUTFILE '/tmp/x'")


def test_blocks_delete_returning_in_subquery() -> None:
    # PG: SELECT * FROM (DELETE ... RETURNING ...) x —— 子查询内含写操作
    with pytest.raises(UnsafeSqlError):
        _guard('SELECT * FROM (DELETE FROM users RETURNING id) x')


# ---------------- 拦截：表白名单 / 命名空间碰撞 ----------------


def test_blocks_system_table_via_namespace_collision() -> None:
    # 关键回归：mysql.user 经命名空间剥离不得与白名单内的 public.user 碰撞放行
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT authentication_string FROM mysql.user')


def test_blocks_information_schema() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT * FROM information_schema.tables')


def test_blocks_non_allowlisted_table() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT * FROM secrets')


def test_blocks_union_exfil_from_non_allowlisted() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT id FROM users UNION SELECT password FROM sys_user')


def test_blocks_non_allowlisted_table_inside_cte() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('WITH active AS (SELECT * FROM secrets) SELECT * FROM active')


def test_explicit_schema_must_match() -> None:
    # 即使表名在白名单，schema 不是 public 也应拒（default_schema=public）
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT * FROM other.users')


# ---------------- 拦截：无表侦察 / 危险函数 ----------------


@pytest.mark.parametrize(
    ['sql'],
    [
        ['SELECT @@hostname'],
        ['SELECT @@version'],
        ['SELECT @@datadir'],
        ['SELECT USER()'],
        ['SELECT CURRENT_USER()'],
        ['SELECT VERSION()'],
        ['SELECT CONNECTION_ID()'],
        ['SELECT 1'],
        ['SELECT 1 + 1'],
    ],
)
def test_blocks_tableless_recon(sql: str) -> None:
    with pytest.raises(UnsafeSqlError):
        _guard(sql)


def test_blocks_dangerous_func_even_with_table() -> None:
    # 即便引用了白名单表，危险函数也要拒
    with pytest.raises(UnsafeSqlError):
        _guard('SELECT SLEEP(5) FROM users')
    with pytest.raises(UnsafeSqlError):
        _guard("SELECT LOAD_FILE('/etc/passwd') FROM users")


def test_blocks_user_variable() -> None:
    with pytest.raises(UnsafeSqlError):
        _guard('SELECT @x := 1 FROM users')


# ---------------- LIMIT 钳制 ----------------


def test_clamps_large_limit() -> None:
    sql = _guard('SELECT * FROM users LIMIT 1000000')
    assert 'LIMIT 100' in sql.upper()
    assert '1000000' not in sql


def test_keeps_limit_below_cap() -> None:
    sql = _guard('SELECT * FROM users LIMIT 10')
    assert 'LIMIT 10' in sql.upper()


# ---------------- 其它 ----------------


def test_blocks_empty_sql() -> None:
    with pytest.raises(UnsafeSqlError):
        _guard('   ')
