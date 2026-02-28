drop table if exists ai_model;
drop table if exists ai_provider;
drop table if exists sys_mcp;

select setval(pg_get_serial_sequence('sys_menu', 'id'), coalesce(max(id), 0) + 1, true) from sys_menu;
