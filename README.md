# AI

为系统提供 AI 赋能

- 基于 AG-UI 的流式生成能力，支持文本与图片生成
- 支持对话列表、详情、重命名、置顶、删除，以及上下文清理
- 支持消息编辑保存、删除、清空，以及基于用户消息或 AI 回复重生成
- 支持默认模型、快捷短语、供应商、模型、MCP 管理，以及批量同步供应商模型
- 支持 MCP、联网搜索、思考参数、图片生成参数、内置工具能力透传，并适配多种供应商类型

## 插件类型

- 应用级插件

## 配置说明

插件目录下 `plugin.toml` 的 `[settings]` 中包含以下内容：

```toml
[settings]
AI_CODE_MODE_TOOLS = []
AI_CODE_MODE_MAX_RETRIES = 3
AI_CODE_MODE_DYNAMIC_CATALOG = false
AI_HTTP_MAX_RETRIES = 5
AI_MCP_MAX_RETRIES = 1
```

当前项目的 `backend/core/conf.py` 已包含以下字段：

```python
##################################################
# [ Plugin ] ai
##################################################
AI_CONFIG_STATUS: bool = True
AI_EXA_API_KEY: str | None = None
AI_TAVILY_API_KEY: str | None = None

# 基础配置（in plugin.toml）
AI_CODE_MODE_TOOLS: list[str]
AI_CODE_MODE_MAX_RETRIES: int
AI_CODE_MODE_DYNAMIC_CATALOG: bool
AI_HTTP_MAX_RETRIES: int
AI_MCP_MAX_RETRIES: int
```

## 使用方式

1. 安装并启用参数配置插件和 AI 插件后，重启后端服务
2. 通过 AI 配置管理菜单维护 `AI_EXA_API_KEY` 和 `AI_TAVILY_API_KEY`
3. 先创建 AI 供应商，再同步或创建对应模型
4. 配置默认助手模型
5. 配置 MCP 和快捷短语等辅助能力，其中 OpenRouter 模型 ID 需使用 `供应商/模型` 格式
6. 发起对话并维护会话历史

## 后端对话流程

后端对话接口以 AG-UI 作为外部协议，以 Pydantic AI `ModelMessage` 作为内部消息与存储格式。AG-UI 主要在请求入口、流式事件输出、历史快照输出三个位置介入。

```mermaid
flowchart TD
  A["前端发起 POST 请求<br/>AG-UI messages + forwardedProps"] --> B["后端 ChatService 接收请求"]

  B --> C{"当前会话是否有 pending assistant？"}
  C -->|"是"| C1["拒绝请求<br/>避免同一会话并发上下文错乱"]
  C -->|"否"| D["AG-UI adapter.decode_input_messages<br/>AG-UI messages -> Pydantic AI ModelRequest"]

  D --> E["AG-UI adapter.build_run_context<br/>提取 prompt / 附件 / 模型参数 / forwardedProps"]
  E --> F["AG-UI adapter.sanitize_input_messages<br/>清洗本次输入消息"]

  F --> G["读取历史消息<br/>DB 中的 Pydantic AI ModelMessage JSON"]
  G --> H["组装 Agent 上下文<br/>history ModelMessage + current ModelRequest"]

  H --> I["开启数据库事务"]
  I --> J["写入 user 消息<br/>status = success<br/>model_messages = Pydantic AI ModelRequest"]
  J --> K["写入 assistant 占位消息<br/>status = pending<br/>model_messages = 空或占位"]
  K --> L["提交事务"]

  L --> M["Pydantic AI Agent.run_stream<br/>开始模型流式推理"]

  M --> N{"流式事件来源"}
  N -->|"文本增量"| O["Pydantic AI text delta"]
  N -->|"思考 / reasoning"| P["Pydantic AI reasoning part"]
  N -->|"工具调用"| Q["Pydantic AI tool call / tool result"]
  N -->|"生命周期"| R["run / step / state"]

  O --> S["AG-UI adapter.build_streaming_response<br/>转换为 TEXT_MESSAGE_*"]
  P --> T["AG-UI adapter.build_streaming_response<br/>转换为 REASONING_* / THINKING_*"]
  Q --> U["AG-UI adapter.build_streaming_response<br/>转换为 TOOL_CALL_* / file event"]
  R --> V["AG-UI adapter.build_streaming_response<br/>转换为 RUN / STEP / STATE"]

  S --> W["SSE 返回前端<br/>AG-UI Event Stream"]
  T --> W
  U --> W
  V --> W

  M --> X{"流结束 / 报错 / 中止"}
  X -->|"成功"| Y["更新 assistant 占位消息<br/>status = success<br/>model_messages = Pydantic AI ModelResponse"]
  X -->|"失败"| Z["更新 assistant 占位消息<br/>status = error<br/>写入错误响应或错误文本"]
  X -->|"中止"| AA["结束流式状态<br/>按当前策略保留 pending 或转 error"]

  Y --> AB["后端完成请求"]
  Z --> AB
  AA --> AB

  AB --> AC["前端收到流结束<br/>合并 transientMessages / 同步会话状态"]

  AD["历史消息接口 / 会话快照"] --> AE["读取 DB<br/>Pydantic AI ModelMessage JSON"]
  AE --> AF["AG-UI adapter.serialize_messages_to_snapshot<br/>Pydantic AI -> AG-UI snapshot"]
  AF --> AG["返回前端历史消息<br/>AG-UI messages"]
```

## 卸载说明

- 卸载插件后，建议同步移除参数配置中的 AI 相关配置和 `backend/core/conf.py` 中的插件配置
- 如前端页面或业务流程已依赖 AI 对话、默认模型、模型、供应商、MCP 等能力，请同步清理对应集成

## 联系方式

- 作者：`wu-clan`
- 反馈方式：提交 Issue 或 PR
