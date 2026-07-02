# AI Resource Policy Roadmap

## 当前设计

- `ai` 插件只提供通用策略扩展点，不依赖任何具体策略插件
- 未注册策略时，资源列表默认全部可见，AI 调用默认全部允许
- 可选插件通过 `hooks.py` 注册策略实现，多个策略可以同时生效
- 策略分为调用前校验、调用后通知两个阶段
- provider / model / MCP 列表接口不接入策略管线，资源展示能力由具体插件自行扩展
- 可选插件可通过 SQLAlchemy listener 独立实现资源可见性过滤，不需要改动 AI 策略管线
- provider / model 解析与调用前策略校验内置在 `open_chat_session` 的内部解析流程中，外部调用不直接绕过策略入口
- `AIInvocationContext` 面向通用调用生命周期，包含用户、供应商类型、供应商名称、模型、MCP、生成类型和会话信息
- `AIInvocationResult` 标准化 Pydantic AI 用量字段，保留原始结果用于少数深度扩展场景
- 可选策略插件可按用户分组、额度、租户、套餐等维度实现调用权限与用量处理

## 模块职责

- `policy/context.py` 定义调用前后策略上下文
- `policy/base.py` 定义策略基类与可实现阶段
- `policy/registry.py` 负责策略注册、调用前校验和调用后通知
- 新策略插件直接从 `backend.plugin.ai.policy.*` 导入策略能力

## 策略阶段

### 资源列表展示

- AI 核心不为资源列表暴露策略入口
- AI 核心不聚合、计算或理解可见性规则
- 可选插件可监听 SQLAlchemy `do_orm_execute` 事件，并对自己关心的资源模型追加查询条件
- `ai_group` 使用此方式实现分组资源可见性，调用安全边界仍由调用前校验负责

### 调用前校验

- 用于真正发起 AI 调用前的最终兜底
- 所有真实调用控制都应该放在此阶段
- 多个策略按注册顺序依次执行
- 任一策略拒绝，本次调用拒绝
- 适合 `ai_group`、`ai_quota`、`ai_tenant`、`ai_billing` 等插件实现调用控制

### 调用后通知

- 用于记录用量、扣减额度、写账单或审计日志
- 多个策略按注册顺序依次通知
- 优先使用标准化用量字段，只有确实需要供应商原始信息时再读取 `raw_result`
- 当前调用后策略异常只记录日志，不影响已完成的主调用流程

## 后续策略组建议

- 支持策略声明严格程度，例如调用后失败是否阻断响应或触发补偿
- 支持策略优先级，明确分组、租户、套餐、额度之间的执行顺序
- 支持更完整的调用结果摘要，例如图片数量、供应商响应 ID
- 适用插件包括 `ai_group`、`ai_quota`、`ai_tenant`、`ai_billing`

## 后端对话流程

后端对话接口以 AG-UI 作为外部协议，以 Pydantic AI `ModelMessage` 作为内部消息与存储格式。AG-UI 主要在请求入口、流式事件输出、历史快照输出三个位置介入

```mermaid
flowchart TD
  S0["后端启动"] --> S1["加载已启用插件"]
  S1 --> S2["按 depends_on 排序"]
  S2 --> S3["加载 AI 插件<br/>初始化策略管理器"]
  S3 --> S4["加载可选策略插件<br/>例如 ai_group / ai_quota / ai_billing"]
  S4 --> S5["执行 hooks.py<br/>register_ai_resource_policy"]
  S5 --> S6["策略进入 AI 策略管线"]
  S5 --> S7["可选插件注册 SQLAlchemy listener<br/>例如 ai_group 可见性过滤"]

  L0["前端打开资源选择器"] --> L1["provider / model / MCP 列表接口"]
  L1 --> L2["AI 资源模型查询"]
  L2 --> L3{"是否启用可见性 listener"}
  S7 -.-> L3
  L3 -->|"是"| L4["插件注入 SQLAlchemy 过滤条件"]
  L3 -->|"否"| L5["保持 AI 默认查询"]
  L4 --> L6["返回当前用户可见资源"]
  L5 --> L6

  A["前端发起 POST 请求<br/>AG-UI messages + forwardedProps"] --> B["后端 ChatService 接收请求"]

  B --> C["AG-UI adapter.decode_input_messages<br/>AG-UI messages -> Pydantic AI ModelRequest"]
  C --> D["AG-UI adapter.build_run_context<br/>提取 conversation_id / forwardedProps"]
  D --> E["open_chat_session<br/>解析 provider / model"]
  E --> F["构建 AIInvocationContext<br/>user_id / provider_id / model_pk / mcp_ids"]
  F --> G["validate_ai_invocation<br/>调用前策略校验"]
  S6 -.-> G

  G --> H{"遍历策略 before_invoke<br/>是否全部通过"}
  H -->|"否"| H1["拒绝请求<br/>抛出 AuthorizationError / RequestError"]
  H -->|"是"| I["构建 AgentSession 与 Agent"]

  I --> J["AG-UI adapter.sanitize_input_messages<br/>清洗本次输入消息"]
  J --> K["读取历史消息<br/>DB 中的 Pydantic AI ModelMessage JSON"]
  K --> L["组装 Agent 上下文<br/>history ModelMessage + current ModelRequest"]

  L --> M["开启数据库事务"]
  M --> N{"当前会话是否有 pending assistant？"}
  N -->|"是"| N1["拒绝请求<br/>避免同一会话并发上下文错乱"]
  N -->|"否"| O["写入 user 消息<br/>status = success"]
  O --> P["写入 assistant 占位消息<br/>status = pending"]
  P --> Q["提交事务"]

  Q --> R["Pydantic AI Agent.run_stream<br/>开始模型流式推理"]

  R --> T{"流式事件来源"}
  T -->|"文本增量"| U["Pydantic AI text delta"]
  T -->|"思考 / reasoning"| V["Pydantic AI reasoning part"]
  T -->|"工具调用"| W["Pydantic AI tool call / tool result"]
  T -->|"生命周期"| X["run / step / state"]

  U --> Y["AG-UI adapter.build_streaming_response<br/>转换为 TEXT_MESSAGE_*"]
  V --> Z["AG-UI adapter.build_streaming_response<br/>转换为 REASONING_* / THINKING_*"]
  W --> AA["AG-UI adapter.build_streaming_response<br/>转换为 TOOL_CALL_* / file event"]
  X --> AB["AG-UI adapter.build_streaming_response<br/>转换为 RUN / STEP / STATE"]

  Y --> AC["SSE 返回前端<br/>AG-UI Event Stream"]
  Z --> AC
  AA --> AC
  AB --> AC

  R --> AD{"流结束 / 报错 / 中止"}
  AD -->|"成功"| AE["更新 assistant 占位消息<br/>status = success<br/>model_messages = Pydantic AI ModelResponse"]
  AD -->|"失败"| AF["更新 assistant 占位消息<br/>status = error<br/>写入错误响应或错误文本"]
  AD -->|"中止"| AG["结束流式状态<br/>按当前策略保留 pending 或转 error"]

  AE --> AH["notify_ai_invocation_result<br/>调用后策略通知"]
  AH --> AI["遍历策略 after_invoke<br/>额度扣减 / 账单 / 审计"]
  AI --> AJ["后端完成请求"]
  AF --> AJ
  AG --> AJ

  AJ --> AK["前端收到流结束<br/>合并 transientMessages / 同步会话状态"]

  AL["历史消息接口 / 会话快照"] --> AM["读取 DB<br/>Pydantic AI ModelMessage JSON"]
  AM --> AN["AG-UI adapter.serialize_messages_to_snapshot<br/>Pydantic AI -> AG-UI snapshot"]
  AN --> AO["返回前端历史消息<br/>AG-UI messages"]
```
