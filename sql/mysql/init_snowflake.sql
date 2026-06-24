insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values (2147098509659213824, 'AI', 'PluginAI', '/plugins/ai', 11, 'tabler:robot', 0, null, null, 1, 1, 1, '', null, null, now(), null);

insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values
(2147098509659213825, 'AI Chat', 'AIChat', '/plugins/ai/chat', 1, 'ri:chat-ai-line', 1, '/plugins/ai/views/chat/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213839, '默认模型', 'AIDefaultModel', '/plugins/ai/default-model', 2, 'carbon:model-alt', 1, '/plugins/ai/views/default-model/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213826, '模型服务', 'AIModelService', '/plugins/ai/model-service', 3, 'carbon:model-alt', 1, '/plugins/ai/views/model-service/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213840, '设置默认模型', 'EditAIDefaultModel', null, 0, null, 2, null, 'ai:default-model:edit', 1, 0, 1, '', null, 2147098509659213839, now(), null),
(2147098509659213827, '新增供应商', 'AddAIProvider', null, 0, null, 2, null, 'ai:provider:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213828, '修改供应商', 'EditAIProvider', null, 0, null, 2, null, 'ai:provider:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213829, '删除供应商', 'DeleteAIProvider', null, 0, null, 2, null, 'ai:provider:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213831, '新增模型', 'AddAIModel', null, 0, null, 2, null, 'ai:model:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213832, '修改模型', 'EditAIModel', null, 0, null, 2, null, 'ai:model:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213833, '删除模型', 'DeleteAIModel', null, 0, null, 2, null, 'ai:model:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213834, '快捷短语', 'AIQuickPhraseManage', '/plugins/ai/quick-phrase', 4, 'mdi:lightning-bolt-outline', 1, '/plugins/ai/views/quick-phrase/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213835, 'MCP 管理', 'AIMcpManage', '/plugins/ai/mcp', 5, 'simple-icons:modelcontextprotocol', 1, '/plugins/ai/views/mcp/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213841, '配置管理', 'AIConfigManage', '/plugins/ai/config', 6, 'codicon:symbol-parameter', 1, '/plugins/ai/views/config/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213842, '新增快捷短语', 'AddAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:add', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213843, '修改快捷短语', 'EditAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:edit', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213844, '删除快捷短语', 'DeleteAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:del', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213836, '新增MCP', 'AddAIMcp', null, 0, null, 2, null, 'ai:mcp:add', 1, 0, 1, '', null, 2147098509659213835, now(), null),
(2147098509659213837, '修改MCP', 'EditAIMcp', null, 0, null, 2, null, 'ai:mcp:edit', 1, 0, 1, '', null, 2147098509659213835, now(), null),
(2147098509659213838, '删除MCP', 'DeleteAIMcp', null, 0, null, 2, null, 'ai:mcp:del', 1, 0, 1, '', null, 2147098509659213835, now(), null),
(2147098509659213845, '保存配置', 'EditAIConfig', null, 0, null, 2, null, 'sys.config.edits', 1, 0, 1, '', null, 2147098509659213841, now(), null);

insert into sys_config (id, name, type, `key`, value, is_frontend, remark, created_time, updated_time)
values
(2147098509659213846, '状态', 'AI', 'AI_CONFIG_STATUS', '1', false, null, now(), null),
(2147098509659213847, 'Exa API Key', 'AI', 'AI_EXA_API_KEY', '', false, '用于 AI 联网搜索的 Exa API Key', now(), null),
(2147098509659213848, 'Tavily API Key', 'AI', 'AI_TAVILY_API_KEY', '', false, '用于 AI 联网搜索的 Tavily API Key', now(), null);
