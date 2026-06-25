delete from sys_menu
where name in (
    'AIChat',
    'AIDefaultModel',
    'EditAIDefaultModel',
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
    'DeleteAIText2SqlDataset',
    'AIConfigManage',
    'EditAIConfig'
);

delete from sys_menu where name = 'PluginAI';

delete from sys_config
where "key" in (
    'AI_CONFIG_STATUS',
    'AI_EXA_API_KEY',
    'AI_TAVILY_API_KEY'
);

drop table if exists ai_message;
drop table if exists ai_conversation;
drop table if exists ai_quick_phrase;
drop table if exists ai_default_model;
drop table if exists ai_model;
drop table if exists ai_provider;
drop table if exists ai_mcp;

drop table if exists ai_text2sql_history;
drop table if exists ai_text2sql_example;
drop table if exists ai_text2sql_table;
drop table if exists ai_text2sql_dataset;

select setval(pg_get_serial_sequence('sys_menu', 'id'), coalesce(max(id), 0) + 1, true) from sys_menu;
