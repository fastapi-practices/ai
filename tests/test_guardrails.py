"""Text2SQL 安全护栏单测（纯逻辑，无 DB）

覆盖：放行普通 SELECT/JOIN/CTE/子查询/已有 LIMIT；拦截非 SELECT、多语句、INTO、越表（含 CTE 体/UNION 内越表）。
"""

import pytest

from backend.plugin.ai.text2sql.exceptions import TableNotAllowedError, UnsafeSqlError
from backend.plugin.ai.text2sql.guardrails import validate_and_normalize

ALLOWLIST = {'users', 'orders', 'products'}


def _guard(sql: str, **kwargs: object) -> str:
    allowlist = kwargs.get('allowlist', ALLOWLIST)  # type: ignore[arg-type]
    max_rows = kwargs.get('max_rows', 100)  # type: ignore[arg-type]
    return validate_and_normalize(sql, allowlist=allowlist, max_rows=max_rows)


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


def test_preserves_existing_limit() -> None:
    sql = _guard('SELECT * FROM users LIMIT 5')
    assert sql.upper().count('LIMIT') == 1


def test_accepts_lowercase_select() -> None:
    sql = _guard('select id from users')
    assert 'LIMIT 100' in sql.upper()


def test_accepts_multiline_and_comment() -> None:
    sql = _guard('/* get users */\nSELECT\n  id\nFROM\n  users\nWHERE 1 = 1')
    assert 'LIMIT 100' in sql.upper()


def test_uses_custom_max_rows() -> None:
    sql = _guard('SELECT * FROM users', max_rows=42)
    assert 'LIMIT 42' in sql.upper()


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


def test_blocks_select_into() -> None:
    with pytest.raises(UnsafeSqlError):
        _guard('SELECT * INTO new_tbl FROM users')


def test_blocks_union_exfil_from_non_allowlisted() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT id FROM users UNION SELECT password FROM sys_user')


def test_blocks_non_allowlisted_table_inside_cte() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('WITH active AS (SELECT * FROM secrets) SELECT * FROM active')


def test_blocks_non_allowlisted_table() -> None:
    with pytest.raises(TableNotAllowedError):
        _guard('SELECT * FROM secrets')


def test_blocks_empty_sql() -> None:
    with pytest.raises(UnsafeSqlError):
        _guard('   ')
