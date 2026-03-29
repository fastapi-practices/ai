delete from sys_menu
where name in (
    'AIChat',
    'AIQuickPhraseManage',
    'AIProviderManage',
    'AddAIProvider',
    'EditAIProvider',
    'DeleteAIProvider',
    'AIModelManage',
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
