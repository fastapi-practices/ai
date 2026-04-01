insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values (2147098509659213824, 'AI', 'PluginAI', '/plugins/ai', 11, 'tabler:robot', 0, null, null, 1, 1, 1, '', null, null, now(), null);

insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values
(2147098509659213825, 'AI Chat', 'AIChat', '/plugins/ai/chat', 1, 'ri:chat-ai-line', 1, '/plugins/ai/views/chat/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213826, '模型服务', 'AIModelService', '/plugins/ai/model-service', 2, 'carbon:model-alt', 1, '/plugins/ai/views/model-service/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213827, '新增供应商', 'AddAIProvider', null, 0, null, 2, null, 'ai:provider:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213828, '修改供应商', 'EditAIProvider', null, 0, null, 2, null, 'ai:provider:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213829, '删除供应商', 'DeleteAIProvider', null, 0, null, 2, null, 'ai:provider:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213831, '新增模型', 'AddAIModel', null, 0, null, 2, null, 'ai:model:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213832, '修改模型', 'EditAIModel', null, 0, null, 2, null, 'ai:model:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213833, '删除模型', 'DeleteAIModel', null, 0, null, 2, null, 'ai:model:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213834, '快捷短语', 'AIQuickPhraseManage', '/plugins/ai/quick-phrase', 3, 'mdi:lightning-bolt-outline', 1, '/plugins/ai/views/quick-phrase/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213835, 'MCP 管理', 'AIMcpManage', '/plugins/ai/mcp', 4, 'simple-icons:modelcontextprotocol', 1, '/plugins/ai/views/mcp/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213836, '新增MCP', 'AddAIMcp', null, 0, null, 2, null, 'ai:mcp:add', 1, 0, 1, '', null, 2147098509659213835, now(), null),
(2147098509659213837, '修改MCP', 'EditAIMcp', null, 0, null, 2, null, 'ai:mcp:edit', 1, 0, 1, '', null, 2147098509659213835, now(), null),
(2147098509659213838, '删除MCP', 'DeleteAIMcp', null, 0, null, 2, null, 'ai:mcp:del', 1, 0, 1, '', null, 2147098509659213835, now(), null);
