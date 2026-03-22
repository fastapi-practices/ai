do $$
declare
    ai_menu_id bigint;
    ai_provider_menu_id bigint;
    ai_model_menu_id bigint;
    ai_mcp_menu_id bigint;
begin
    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values ('AI', 'PluginAI', '/ai', 11, 'tabler:robot', 0, null, null, 1, 1, 1, '', null, null, now(), null)
    returning id into ai_menu_id;

    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values ('AI Chat', 'AIChat', '/ai/chat', 1, 'ri:chat-ai-line', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, ai_menu_id, now(), null);

    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values ('供应商管理', 'AIProviderManage', '/ai/provider', 2, 'mdi:hub-outline', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, ai_menu_id, now(), null)
    returning id into ai_provider_menu_id;

    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values ('模型管理', 'AIModelManage', '/ai/model', 3, 'carbon:model-alt', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, ai_menu_id, now(), null)
    returning id into ai_model_menu_id;

    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values ('MCP 管理', 'AIMcpManage', '/ai/mcp', 4, 'simple-icons:modelcontextprotocol', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, ai_menu_id, now(), null)
    returning id into ai_mcp_menu_id;

    insert into sys_menu (title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
    values
    ('新增供应商', 'AddAIProvider', null, 0, null, 2, null, 'ai:provider:add', 1, 0, 1, '', null, ai_provider_menu_id, now(), null),
    ('修改供应商', 'EditAIProvider', null, 0, null, 2, null, 'ai:provider:edit', 1, 0, 1, '', null, ai_provider_menu_id, now(), null),
    ('删除供应商', 'DeleteAIProvider', null, 0, null, 2, null, 'ai:provider:del', 1, 0, 1, '', null, ai_provider_menu_id, now(), null),
    ('新增模型', 'AddAIModel', null, 0, null, 2, null, 'ai:model:add', 1, 0, 1, '', null, ai_model_menu_id, now(), null),
    ('修改模型', 'EditAIModel', null, 0, null, 2, null, 'ai:model:edit', 1, 0, 1, '', null, ai_model_menu_id, now(), null),
    ('删除模型', 'DeleteAIModel', null, 0, null, 2, null, 'ai:model:del', 1, 0, 1, '', null, ai_model_menu_id, now(), null),
    ('新增MCP', 'AddAIMcp', null, 0, null, 2, null, 'ai:mcp:add', 1, 0, 1, '', null, ai_mcp_menu_id, now(), null),
    ('修改MCP', 'EditAIMcp', null, 0, null, 2, null, 'ai:mcp:edit', 1, 0, 1, '', null, ai_mcp_menu_id, now(), null),
    ('删除MCP', 'DeleteAIMcp', null, 0, null, 2, null, 'ai:mcp:del', 1, 0, 1, '', null, ai_mcp_menu_id, now(), null);
end $$;

select setval(pg_get_serial_sequence('sys_menu', 'id'), coalesce(max(id), 0) + 1, true) from sys_menu;
