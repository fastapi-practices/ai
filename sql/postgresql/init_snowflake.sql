insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values (2147098509659213824, 'AI', 'PluginAI', '/ai', 11, 'tabler:robot', 0, null, null, 1, 1, 1, '', null, null, now(), null);

insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values
(2147098509659213825, 'AI Chat', 'AIChat', '/ai/chat', 1, 'ri:chat-ai-line', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213826, '供应商管理', 'AIProviderManage', '/ai/provider', 2, 'mdi:hub-outline', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213827, '新增供应商', 'AddAIProvider', null, 0, null, 2, null, 'ai:provider:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213828, '修改供应商', 'EditAIProvider', null, 0, null, 2, null, 'ai:provider:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213829, '删除供应商', 'DeleteAIProvider', null, 0, null, 2, null, 'ai:provider:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213830, '模型管理', 'AIModelManage', '/ai/model', 3, 'carbon:model-alt', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213831, '新增模型', 'AddAIModel', null, 0, null, 2, null, 'ai:model:add', 1, 0, 1, '', null, 2147098509659213830, now(), null),
(2147098509659213832, '修改模型', 'EditAIModel', null, 0, null, 2, null, 'ai:model:edit', 1, 0, 1, '', null, 2147098509659213830, now(), null),
(2147098509659213833, '删除模型', 'DeleteAIModel', null, 0, null, 2, null, 'ai:model:del', 1, 0, 1, '', null, 2147098509659213830, now(), null),
(2147098509659213834, 'MCP 管理', 'AIMcpManage', '/ai/mcp', 4, 'simple-icons:modelcontextprotocol', 1, '/plugins/ai/views/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213835, '新增MCP', 'AddAIMcp', null, 0, null, 2, null, 'ai:mcp:add', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213836, '修改MCP', 'EditAIMcp', null, 0, null, 2, null, 'ai:mcp:edit', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213837, '删除MCP', 'DeleteAIMcp', null, 0, null, 2, null, 'ai:mcp:del', 1, 0, 1, '', null, 2147098509659213834, now(), null);
