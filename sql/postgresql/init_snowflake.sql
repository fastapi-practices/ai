insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values (2147098509659213824, 'ai.menu', 'PluginAI', '/plugins/ai', 11, 'tabler:robot', 0, null, null, 1, 1, 1, '', null, null, now(), null);

insert into sys_menu (id, title, name, path, sort, icon, type, component, perms, status, display, cache, link, remark, parent_id, created_time, updated_time)
values
(2147098509659213825, 'ai.chat', 'AIChat', '/plugins/ai/chat', 1, 'ri:chat-ai-line', 1, '/plugins/ai/views/chat/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213839, 'ai.default-model', 'AIDefaultModel', '/plugins/ai/default-model', 2, 'carbon:model-alt', 1, '/plugins/ai/views/default-model/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213826, 'ai.model-service', 'AIModelService', '/plugins/ai/model-service', 3, 'carbon:model-alt', 1, '/plugins/ai/views/model-service/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213840, '设置默认模型', 'EditAIDefaultModel', null, 0, null, 2, null, 'ai:default-model:edit', 1, 0, 1, '', null, 2147098509659213839, now(), null),
(2147098509659213827, '新增供应商', 'AddAIProvider', null, 0, null, 2, null, 'ai:provider:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213828, '修改供应商', 'EditAIProvider', null, 0, null, 2, null, 'ai:provider:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213829, '删除供应商', 'DeleteAIProvider', null, 0, null, 2, null, 'ai:provider:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213831, '新增模型', 'AddAIModel', null, 0, null, 2, null, 'ai:model:add', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213832, '修改模型', 'EditAIModel', null, 0, null, 2, null, 'ai:model:edit', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213833, '删除模型', 'DeleteAIModel', null, 0, null, 2, null, 'ai:model:del', 1, 0, 1, '', null, 2147098509659213826, now(), null),
(2147098509659213834, 'ai.quick-phrase', 'AIQuickPhraseManage', '/plugins/ai/quick-phrase', 4, 'mdi:lightning-bolt-outline', 1, '/plugins/ai/views/quick-phrase/index', null, 1, 1, 1, '', null, 2147098509659213824, now(), null),
(2147098509659213842, '新增快捷短语', 'AddAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:add', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213843, '修改快捷短语', 'EditAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:edit', 1, 0, 1, '', null, 2147098509659213834, now(), null),
(2147098509659213844, '删除快捷短语', 'DeleteAIQuickPhrase', null, 0, null, 2, null, 'ai:quick-phrase:del', 1, 0, 1, '', null, 2147098509659213834, now(), null);
