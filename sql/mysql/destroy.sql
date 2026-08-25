delete from sys_menu
where name in (
    'AIChat',
    'AIDefaultModel',
    'EditAIDefaultModel',
    'AIModelService',
    'AddAIProvider',
    'EditAIProvider',
    'DeleteAIProvider',
    'AddAIModel',
    'EditAIModel',
    'DeleteAIModel',
    'AIQuickPhraseManage',
    'AddAIQuickPhrase',
    'EditAIQuickPhrase',
    'DeleteAIQuickPhrase'
);

delete from sys_menu where name = 'PluginAI';

drop table if exists ai_message;
drop table if exists ai_conversation;
drop table if exists ai_quick_phrase;
drop table if exists ai_default_model;
drop table if exists ai_model;
drop table if exists ai_provider;
