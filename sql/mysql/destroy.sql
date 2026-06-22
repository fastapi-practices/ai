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
    'AddAIQuickPhrase',
    'EditAIQuickPhrase',
    'DeleteAIQuickPhrase',
    'AIMcpManage',
    'AddAIMcp',
    'EditAIMcp',
    'DeleteAIMcp',
    'AIText2SqlDataset',
    'AddAIText2SqlDataset',
    'EditAIText2SqlDataset',
    'DeleteAIText2SqlDataset'
);

delete from sys_menu where name = 'PluginAI';

drop table if exists ai_message;
drop table if exists ai_conversation;
drop table if exists ai_quick_phrase;
drop table if exists ai_model;
drop table if exists ai_provider;
drop table if exists ai_mcp;

drop table if exists ai_text2sql_history;
drop table if exists ai_text2sql_example;
drop table if exists ai_text2sql_table;
drop table if exists ai_text2sql_dataset;
