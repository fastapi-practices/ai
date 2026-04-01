delete from sys_menu
where name in (
    'AIChat',
    'AIQuickPhraseManage',
    'AIModelService',
    'AddAIProvider',
    'EditAIProvider',
    'DeleteAIProvider',
    'AddAIModel',
    'EditAIModel',
    'DeleteAIModel',
    'AIMcpManage',
    'AddAIMcp',
    'EditAIMcp',
    'DeleteAIMcp'
);

delete from sys_menu where name = 'PluginAI';

drop table if exists ai_message;
drop table if exists ai_conversation;
drop table if exists ai_quick_phrase;
drop table if exists ai_model;
drop table if exists ai_provider;
drop table if exists ai_mcp;

select setval(pg_get_serial_sequence('sys_menu', 'id'), coalesce(max(id), 0) + 1, true) from sys_menu;
