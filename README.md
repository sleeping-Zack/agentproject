# 智扫通机器人智能客服 Agent

这是一个面向扫地 / 扫拖机器人的 **RAG + 多工具 Agent + Harness 控制层** 项目。项目不仅覆盖知识库问答、天气 / 环境适配、用户设备使用记录查询、个性化报告生成，也把 Agent 应用生产化中常见的控制问题纳入架构：统一状态、预算停止、动态工具策略、真实人工审批、答案验证、artifact 留存、诊断 trace、评测门禁和服务化交付。

一句话概括：

> `ReactAgent` 负责 ReAct 推理和工具调用，`AgentRunner` 负责请求级控制；工具、RAG、MCP、审批、artifact、trace、metrics 都围绕 Harness 形成可治理的 Agent 运行框架。

---

## 1. 功能概览

- **RAG 知识库**：从 `data/` 中的 PDF / TXT 构建 Chroma 向量库，支持检索、hybrid rank、RAG 注入过滤和引用来源。
- **多工具 Agent**：支持知识库检索、天气、用户位置、用户 ID、当前月份、使用记录查询和报告上下文切换。
- **Harness 控制层**：统一 `AgentRunner` / `AgentState`，支持预算停止、动态工具策略、真实审批、答案验证、artifact 存储和诊断 trace。
- **动态工具治理**：`ToolRegistry` 管工具元数据，`ToolPolicy` 按 tenant / role / scene / tool / args 决策 `allow / deny / need_approval / need_redaction`。
- **真实 HITL 审批**：敏感工具如 `fetch_external_data` 会进入 `SQLiteApprovalStore`，普通用户需审批，operator / admin 可审批。
- **答案质量闸门**：`AnswerVerifier` 检查 evidence、引用、空答案，并可接入 `LLMJudge` 评估 correctness / faithfulness / completeness。
- **产物留存**：`SQLiteArtifactStore` 按 request_id 保存 final answer、verification failure、evidence、tool results 等运行产物。
- **服务化入口**：FastAPI 暴露 `/chat`、`/chat/stream`、`/harness/run`、审批、artifact、MCP、trace、metrics、judge 等接口。
- **MCP 工具服务**：支持 JSON-RPC `initialize`、`tools/list`、`tools/call`；MCP 工具调用同样经过 ToolPolicy 和审批存储。
- **可观测性**：包含 request/tool/model trace、diagnostic event、OpenTelemetry 风格 span、Prometheus 文本指标和 SSE 外观。
- **评测门禁**：Agent golden set 支持 pass_rate、tool_recall、keyword_recall、P95 延迟、平均成本和失败 bucket 拆解。

---

## 2. 环境与启动

推荐 Python 3.10.x，本项目使用 Python 3.10.11 验证。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 中配置 `DASHSCOPE_API_KEY`。

加载知识库：

```powershell
python -m rag.vector_store
```

启动 Streamlit 演示：

```powershell
streamlit run app.py
```

启动 FastAPI 服务：

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

启动 MCP stdio server：

```powershell
python mcp_server.py
```

---

## 3. 常用接口

| 接口 | 作用 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /tools/manifest` | 导出工具 manifest |
| `POST /chat` | 兼容聊天入口，内部走 Harness |
| `POST /chat/stream` | SSE 外观接口，返回 harness 最终结果 |
| `POST /harness/run` | 推荐生产入口，返回 status / approval_id / verifier / artifacts |
| `GET /approvals/{approval_id}` | 查询审批记录 |
| `POST /approvals/{approval_id}/approve` | 审批通过，仅 operator / admin |
| `POST /approvals/{approval_id}/deny` | 审批拒绝，仅 operator / admin |
| `GET /artifacts/{request_id}` | 按 request_id 查询产物 |
| `GET /artifact/{artifact_id}` | 查询单个 artifact |
| `POST /plan` | Planner 多任务接口 |
| `POST /judge` | LLM-as-Judge 批量评测接口 |
| `POST /mcp` | HTTP JSON-RPC MCP 入口 |
| `GET /metrics` | Prometheus 文本指标 |
| `GET /metrics/snapshot` | JSON 指标快照 |
| `GET /traces/{request_id}` | 查询 trace payload |
| `GET /traces/{request_id}/otel` | 导出 OpenTelemetry 风格 spans |

入口关系：

- `/harness/run` 是推荐的生产控制入口。
- `/chat` 保留兼容旧调用方，但内部已经调用 `AgentRunner`，不会绕过审批、artifact、trace 和 verifier。
- `/chat/stream` 是 SSE 外观接口，当前以 harness 最终结果发送 `answer/done` 事件，不再直接暴露旧 ReAct stream。
- `/mcp tools/call` 由 `MCPToolServer` 执行 ToolPolicy；调用 `fetch_external_data` 等敏感工具时会返回 `pending_approval` 和 `approval_id`，审批通过且参数匹配后才执行。
- `user_role` 不信任 request body，服务端从 `X-User-Role` 等 auth header 中解析；`approve/deny` 需要 `operator` 或 `admin`。

---

## 4. 最新总体架构图

```mermaid
flowchart TB
    User["用户 / 前端 / 外部系统"] --> Streamlit["Streamlit UI\napp.py"]
    User --> FastAPI["FastAPI 服务\napi/server.py"]
    User --> MCPClient["MCP Client"]

    Streamlit --> ReactAgent["ReactAgent\nReAct 执行面"]

    FastAPI --> Auth["API Key 鉴权\nresolve_auth_context"]
    Auth --> Tenant["tenant / role / principal\n可信上下文"]
    Tenant --> Rate["RateLimiter\n租户或 IP 限流"]
    Rate --> Safe["Prompt Injection 检测"]

    Safe --> Chat["/chat"]
    Safe --> Stream["/chat/stream"]
    Safe --> HarnessAPI["/harness/run"]
    Safe --> PlanAPI["/plan"]
    Safe --> MCPHTTP["/mcp"]

    Chat --> Runner["AgentRunner\nHarness 控制面"]
    Stream --> Runner
    HarnessAPI --> Runner

    Runner --> State["AgentState\n状态机"]
    Runner --> Budget["Budget\nsteps / tools / tokens / cost"]
    Runner --> Policy["ToolPolicy\n动态工具策略"]
    Runner --> Approval["SQLiteApprovalStore\nHITL 审批"]
    Runner --> Backend["ReactAgentBackend\n后端适配器"]
    Runner --> Verifier["AnswerVerifier\n答案验证"]
    Runner --> Artifact["SQLiteArtifactStore\n产物留存"]
    Runner --> Diag["Diagnostic Trace\n诊断事件"]

    Backend --> ReactAgent
    ReactAgent --> LC["LangChain create_agent\nReAct 工具调用"]
    LC --> Tools["Agent Tools\n工具函数层"]

    Tools --> Registry["ToolRegistry\nmanifest / scope / risk"]
    Tools --> Middleware["Tool Middleware\npolicy / budget / cache / retry / breaker / trace"]
    Tools --> RAG["RagSummarizeService"]
    Tools --> Data["ToolDataService\n配置 / CSV 数据"]

    RAG --> Vector["VectorStoreService\nChroma"]
    Vector --> KB["data/ PDF TXT 知识库"]
    RAG --> Model["Chat Model"]
    Model --> Router["Provider / ModelRouter"]

    ReactAgent --> Memory["ConversationMemory"]
    Memory --> SessionDB["SQLiteStore\nsession_messages"]

    FastAPI --> Trace["TraceRecorder"]
    Trace --> TraceDB["SQLiteStore\ntraces"]
    FastAPI --> Metrics["MetricsRegistry"]

    MCPClient --> MCPStdio["mcp_server.py"]
    MCPStdio --> MCPServer["MCPToolServer"]
    MCPHTTP --> MCPServer
    MCPServer --> Policy
    MCPServer --> Approval
    MCPServer --> Tools

    PlanAPI --> Planner["PlannerAgent\nplan -> execute -> aggregate"]
    Planner --> Tools
```

核心分工：

- **执行面**：`ReactAgent`、LangChain ReAct、工具、RAG、数据服务。
- **控制面**：`AgentRunner`、`AgentState`、`Budget`、`ToolPolicy`、审批、Verifier、Artifact、Diagnostic Trace。
- **接入面**：FastAPI、Streamlit、MCP stdio / HTTP。
- **治理面**：鉴权、限流、安全、metrics、trace、评测门禁。

---

## 5. 请求入口与可信上下文架构

```mermaid
flowchart LR
    Req["HTTP 请求"] --> APIKey["X-API-Key\n基础鉴权"]
    APIKey --> AuthCtx["resolve_auth_context"]
    AuthCtx --> Tenant["tenant_id\nX-Tenant-ID 或 body tenant_id"]
    AuthCtx --> Role["user_role\nX-User-Role"]
    AuthCtx --> Principal["principal_id\nX-Principal-ID"]
    Role --> Validate["VALID_ROLES\nuser / operator / admin"]
    Validate --> Rate["RateLimiter"]
    Rate --> Safe["assert_safe_user_input"]
    Safe --> Task["AgentTask"]
```

说明：

- `X-API-Key` 只负责基础 API 鉴权。
- `AuthContext` 统一解析 `tenant_id`、`user_role`、`principal_id`。
- `user_role` 不从 request body 中信任读取；生产环境应替换成 API Key / JWT claims。
- `operator` 和 `admin` 属于审批角色，可调用 approve / deny。
- 请求进入 Harness 前仍会经过限流和 Prompt Injection 检测。

---

## 6. Harness 控制层架构

```mermaid
flowchart TB
    Task["AgentTask\nquery/session/tenant/role/scene/approval_id"] --> Runner["AgentRunner.run"]

    Runner --> TraceStart["ensure_trace"]
    TraceStart --> Scene["resolve_scene\n默认 / report"]
    Scene --> State["创建 AgentState"]
    State --> BudgetCheck["Budget.can_continue"]

    BudgetCheck -->|超预算| Blocked["status=blocked"]
    BudgetCheck -->|未超预算| Sensitive["handle_sensitive_report_data"]

    Sensitive --> Policy["ToolPolicy.decide"]
    Policy -->|deny| Rejected["status=rejected"]
    Policy -->|need_approval| Pending["SQLiteApprovalStore\nstatus=pending"]
    Policy -->|allow 或 approved| Backend["ReactAgentBackend"]

    Pending --> ReturnPending["返回 pending_approval\napproval_id"]

    Backend --> Usage["抽取 evidence / tool_results / usage"]
    Usage --> BudgetRecord["record_tokens / record_cost / record_tool_call"]
    BudgetRecord --> Verifier["AnswerVerifier.verify"]

    Verifier -->|retry| Backend
    Verifier -->|refuse| FailArtifact["保存 verification_failure artifact"]
    Verifier -->|accept| SaveArtifact["保存 final-answer artifact"]

    SaveArtifact --> Completed["status=completed"]
    FailArtifact --> Rejected
    Blocked --> Result["AgentRunResult"]
    Rejected --> Result
    Completed --> Result
    ReturnPending --> Result
```

`AgentRunner` 的关键价值：

| 控制职责 | 对应模块 |
|---|---|
| 统一运行输入 | `AgentTask` |
| 统一运行状态 | `AgentState` |
| 防止无限调用 | `Budget` |
| 动态权限裁决 | `ToolPolicy` |
| 敏感工具人工审批 | `SQLiteApprovalStore` |
| 复用旧 ReAct 执行 | `ReactAgentBackend` |
| 输出质量闸门 | `AnswerVerifier` |
| 结果留存 | `SQLiteArtifactStore` |
| 诊断追踪 | `TraceRecorder.record_diagnostic_event` |

---

## 7. Harness 时序图

```mermaid
sequenceDiagram
    participant User as 用户/客户端
    participant API as FastAPI
    participant Auth as AuthContext
    participant Runner as AgentRunner
    participant Policy as ToolPolicy
    participant Approval as SQLiteApprovalStore
    participant Backend as ReactAgentBackend
    participant Agent as ReactAgent
    participant Verifier as AnswerVerifier
    participant Artifact as SQLiteArtifactStore
    participant Trace as TraceRecorder

    User->>API: POST /harness/run 或 /chat
    API->>Auth: API Key + tenant + role + principal
    API->>API: rate limit + prompt injection check
    API->>Runner: AgentTask
    Runner->>Trace: ensure_trace + diagnostic
    Runner->>Runner: create AgentState + Budget
    Runner->>Policy: decide(fetch_external_data when report)

    alt 需要审批
        Policy-->>Runner: NEED_APPROVAL
        Runner->>Approval: create_pending
        Runner->>Trace: diagnostic approval pending
        Runner-->>API: pending_approval + approval_id
        API-->>User: 等待审批
    else 允许或已审批
        Policy-->>Runner: ALLOW
        Runner->>Backend: execute_agent_backend
        Backend->>Agent: execute_stream(query, context)
        Agent-->>Backend: final answer + trace events
        Backend-->>Runner: AgentBackendResult
        Runner->>Verifier: verify(answer, evidence, tools)
        alt 验证通过
            Runner->>Artifact: save final-answer
            Runner->>Trace: diagnostic completed
            Runner-->>API: completed + answer + artifact
        else 验证失败
            Runner->>Artifact: save verification_failure
            Runner->>Trace: diagnostic failed
            Runner-->>API: rejected
        end
    end
```

---

## 8. AgentState 状态机

```mermaid
stateDiagram-v2
    [*] --> running
    running --> pending_approval: need_approval
    pending_approval --> running: approved + rerun
    pending_approval --> rejected: denied
    running --> blocked: budget_exhausted
    running --> failed: backend_exception
    running --> rejected: policy_denied / verifier_failed
    running --> completed: verifier_passed
    completed --> [*]
    rejected --> [*]
    blocked --> [*]
    failed --> [*]
```

`AgentState` 统一保存：

- `request_id / session_id / tenant_id`
- `user_goal / user_role / scene`
- `steps / observations / tool_calls / artifacts`
- `budget / status / final_answer / error / approval_id`

状态含义：

| 状态 | 含义 |
|---|---|
| `running` | 正在执行 |
| `pending_approval` | 等待人工审批 |
| `blocked` | 被预算或前置控制阻断 |
| `failed` | 后端异常失败 |
| `rejected` | 策略或验证拒绝输出 |
| `completed` | 正常完成 |

---

## 9. 策略与审批架构

```mermaid
flowchart TB
    ToolCall["候选工具调用"] --> Registry["ToolRegistry.get_spec"]
    Registry --> Allowed{"是否注册且在 allowlist"}
    Allowed -->|否| Deny["DENY"]
    Allowed -->|是| Args["检查参数字段"]

    Args --> Secret{"包含 token / secret / api_key"}
    Secret -->|是| Redact["NEED_REDACTION"]
    Secret -->|否| Sensitive{"是否敏感工具"}

    Sensitive -->|否| ApprovalMeta{"requires_approval"}
    Sensitive -->|是| Scene{"scene 是否 report/monthly_report"}
    Scene -->|否| Deny
    Scene -->|是| Role{"role 是否 operator/admin"}
    Role -->|是| Allow["ALLOW"]
    Role -->|否| NeedApproval["NEED_APPROVAL"]

    ApprovalMeta -->|否| Allow
    ApprovalMeta -->|是| Role2{"role 是否 operator/admin"}
    Role2 -->|是| Allow
    Role2 -->|否| NeedApproval

    NeedApproval --> ApprovalDB["SQLiteApprovalStore\ncreate_pending"]
    ApprovalDB --> Pending["pending_approval + approval_id"]
    Allow --> Invoke["允许进入 backend/tool"]
```

当前核心敏感工具：

| 工具 | scope | risk_level | side_effect | requires_approval |
|---|---|---|---|---|
| `fetch_external_data` | `usage_record:read` | `medium` | `read_sensitive` | `true` |

审批接口：

```text
GET  /approvals/{approval_id}
POST /approvals/{approval_id}/approve
POST /approvals/{approval_id}/deny
```

审批安全规则：

- 只能查询当前 tenant 下的审批记录。
- approve / deny 需要 `operator` 或 `admin`。
- 审批通过后，Runner 和 Tool Middleware 都会校验 `tenant_id` 与 `tool_name` 是否匹配。

---

## 10. ReactAgent 执行面架构

```mermaid
flowchart TB
    Backend["ReactAgentBackend"] --> Agent["ReactAgent.execute_stream"]
    Agent --> Trace["start_trace / bind context"]
    Agent --> Memory["ConversationMemory\n读取历史"]
    Agent --> LC["LangChain create_agent"]

    LC --> Prompt["main_prompt / report_prompt\ndynamic_prompt"]
    LC --> Model["chat_model"]
    LC --> Tools["tools"]
    LC --> Middleware["middleware"]

    Tools --> RAGTool["rag_summarize"]
    Tools --> Weather["get_weather"]
    Tools --> Location["get_user_location"]
    Tools --> UserID["get_user_id"]
    Tools --> Month["get_current_month"]
    Tools --> External["fetch_external_data"]
    Tools --> ReportCtx["fill_context_for_report"]

    Agent --> MemoryWrite["写入用户与助手消息"]
    Agent --> Chunks["流式 chunks"]
    Chunks --> BackendResult["AgentBackendResult"]
```

说明：

- `ReactAgentBackend` 是 Harness 和旧 ReAct Agent 之间的适配层。
- Backend 会把 `user_role / scene / approval_id / max_tool_calls` 传入 `ReactAgent.execute_stream`。
- 执行后，Backend 从 trace 中抽取 evidence、tool_results、tokens、cost，封装为 `AgentBackendResult`。
- Harness 不重写 LangChain ReAct loop，只在外层提供控制面能力。

---

## 11. 工具治理架构

```mermaid
flowchart LR
    Agent["LangChain Agent"] --> Middleware["Tool Middleware"]

    Middleware --> Allow["ToolRegistry.allowlist"]
    Middleware --> Budget["max_tool_calls"]
    Middleware --> Policy["ToolPolicy"]
    Middleware --> Approval["ApprovalStore"]
    Middleware --> Breaker["CircuitBreaker"]
    Middleware --> Timeout["timeout_seconds"]
    Middleware --> Cache["ToolCallCache"]
    Middleware --> Retry["RetryPolicy"]
    Middleware --> Trace["TraceRecorder\nargs_hash + redacted_args"]
    Middleware --> Event["EventBus\ntool_start/tool_end/approval_required"]

    Cache -->|hit| ReturnCache["返回缓存 ToolMessage"]
    Retry --> RealTool["真实工具函数"]
    RealTool --> RAG["RAG 服务"]
    RealTool --> Data["ToolDataService"]
```

工具中间件的职责：

1. 检查工具是否在 allowlist。
2. 执行工具调用预算控制。
3. 执行 ToolPolicy；需要审批时创建 pending approval 或校验 approval_id。
4. 通过 CircuitBreaker 保护失败工具。
5. 按 `timeout_seconds` 控制工具调用超时。
6. 使用 ToolCallCache 复用幂等工具结果。
7. 用 RetryPolicy 处理偶发失败。
8. trace 中只记录 `args_hash` 和 `redacted_args`，避免泄露敏感参数。
9. 通过 EventBus 发布 `tool_start / tool_end / approval_required`。

---

## 12. RAG 知识库架构

```mermaid
flowchart TB
    Files["data/ 下 PDF / TXT 文件"] --> Loader["pdf_loader / txt_loader"]
    Loader --> MD5["MD5 去重"]
    MD5 --> Splitter["RecursiveCharacterTextSplitter\nchunk_size=200 overlap=20"]
    Splitter --> Meta["metadata\nsource / chunk_version / chunk_index / doc_id"]
    Meta --> Chroma["Chroma 向量库\nstorage/chroma"]

    Query["用户问题"] --> Retriever["Retriever k=3"]
    Chroma --> Retriever
    Retriever --> Hybrid["hybrid_rank\n关键词 + 向量重排"]
    Hybrid --> Safety["RAG 注入过滤"]
    Safety --> Prompt["RAG Prompt\ninput + context"]
    Prompt --> LLM["Chat Model"]
    LLM --> Answer["答案 + 引用来源"]
    Safety --> Evidence["trace rag_evidence\n供 verifier 使用"]
```

RAG 问答由 `RagSummarizeService` 实现：先从 Chroma 检索，再经过 `hybrid_rank` 重排，对检索内容做 RAG 注入检测，最后将 query 和 context 填入 RAG prompt 调用模型，并在答案后附加引用来源。

---

## 13. 报告生成与敏感数据审批链路

```mermaid
flowchart TB
    User["生成本月使用记录报告"] --> API["/chat 或 /harness/run"]
    API --> Runner["AgentRunner"]
    Runner --> Scene["resolve_scene -> report"]
    Scene --> Policy["ToolPolicy 判断 fetch_external_data"]

    Policy -->|user| Pending["pending_approval"]
    Pending --> ApprovalDB["SQLiteApprovalStore"]
    ApprovalDB --> Approve["operator/admin approve"]
    Approve --> Rerun["携带 approval_id 再次调用"]

    Policy -->|operator/admin| Backend["ReactAgentBackend"]
    Rerun --> Backend

    Backend --> Agent["ReactAgent"]
    Agent --> ReportCtx["fill_context_for_report"]
    Agent --> UID["get_user_id"]
    Agent --> Month["get_current_month"]
    Agent --> Record["fetch_external_data"]
    Agent --> RAG["rag_summarize 保养建议"]
    Agent --> Report["结构化报告"]

    Report --> Verifier["AnswerVerifier"]
    Verifier --> Artifact["final-answer artifact"]
    Artifact --> Resp["返回 completed"]
```

设计要点：

- 报告类请求会被识别为 `report` scene。
- 读取用户使用记录属于 `read_sensitive`，普通用户会先进入审批。
- 审批由 operator / admin 完成，且审批记录按 tenant 隔离。
- 审批通过后，`approval_id` 会进入 Agent runtime context，工具中间件再次校验 approval 与当前租户、工具是否匹配。

---

## 14. Planner 多任务架构

```mermaid
flowchart LR
    UserQuery["复杂用户请求"] --> Planner["TaskPlanner\n规则/可替换 LLM 规划"]
    Planner --> Validator["PlanValidator\n类型/依赖/步数校验"]
    Validator --> Tasks["SubTask[]\nweather / rag_qa / report / generic"]
    Tasks --> Executor["PlanExecutor\nThreadPoolExecutor 并发执行"]

    Executor --> H1["weather handler"]
    Executor --> H2["rag_qa handler"]
    Executor --> H3["report handler"]
    Executor --> H4["generic handler"]

    H1 --> Results["SubTaskResult[]"]
    H2 --> Results
    H3 --> Results
    H4 --> Results

    Results --> Aggregator["ResultAggregator"]
    Aggregator --> Final["综合回答"]
    Validator --> Replanner["Replanner\n失败兜底任务"]
```

`agent/planner.py` 实现 `plan -> execute -> aggregate`。`TaskPlanner` 根据关键词拆成 `weather / rag_qa / report / generic` 子任务，`PlanExecutor` 可并发执行无依赖任务，`ResultAggregator` 合并结果。`PlanValidator` 和 `Replanner` 为计划型任务增加保护。

---

## 15. MCP 架构

```mermaid
flowchart TB
    MCPClient["MCP Client"] -->|stdio| Stdio["mcp_server.py"]
    HTTPClient["HTTP Client"] -->|POST /mcp| API["FastAPI /mcp"]

    Stdio --> Server["MCPToolServer"]
    API --> Auth["AuthContext"]
    Auth --> Server

    Server --> Init["initialize"]
    Server --> List["tools/list"]
    Server --> Call["tools/call"]

    List --> Manifest["ToolRegistry.as_mcp_manifest"]
    Call --> Policy["ToolPolicy"]
    Policy -->|allow| Handler["tool_handlers"]
    Policy -->|need_approval| Approval["SQLiteApprovalStore\npending_approval"]
    Handler --> RAG["rag_summarize"]
    Handler --> Weather["get_weather"]
    Handler --> Record["fetch_external_data"]
```

说明：

- `MCPToolServer` 支持 JSON-RPC `initialize`、`tools/list`、`tools/call`。
- `tools/list` 返回带 risk_level、side_effect、requires_approval、timeout_seconds 的 manifest。
- `tools/call` 会经过 ToolPolicy；敏感工具未审批时返回 pending approval，而不是直接调用 handler。
- HTTP `/mcp` 会传入 `AuthContext`，MCP 工具权限与 HTTP Agent 权限保持一致。

---

## 16. 会话记忆与持久化架构

```mermaid
flowchart LR
    Agent["ReactAgent"] --> Memory["ConversationMemory"]
    Memory --> Cache["进程内 cache\n最近消息窗口"]
    Memory --> Summary["ConversationSummarizer\n长对话摘要"]
    Memory --> Store["SessionStore 协议"]
    Store --> SQLite["SQLiteStore"]

    SQLite --> SessionTable["session_messages\ntenant_id + session_id + role + content"]
    SQLite --> TraceTable["traces\nrequest_id + session_id + tenant_id + payload"]

    Runner["AgentRunner"] --> ArtifactStore["SQLiteArtifactStore"]
    ArtifactStore --> ArtifactTable["artifacts\nrequest_id + tenant_id + type + payload"]

    Runner --> ApprovalStore["SQLiteApprovalStore"]
    ApprovalStore --> ApprovalTable["approvals\napproval_id + tenant_id + tool + status"]
```

持久化对象：

| 存储 | 作用 |
|---|---|
| `SQLiteStore` | session messages 与 traces |
| `SQLiteApprovalStore` | pending / approved / denied 审批记录 |
| `SQLiteArtifactStore` | final answer、verification failure、evidence、tool_results 等产物 |

---

## 17. 可观测性架构

```mermaid
flowchart TB
    Request["一次用户请求"] --> Trace["TraceRecorder"]

    Trace --> SpanAgent["agent / model / tool span"]
    Trace --> Diag["diagnostic event"]
    Diag --> Fields["step_id / status / tool / args_hash / tokens / cost / evidence_ids / verifier / retry / prompt_version / model_name / failure_reason"]

    Trace --> Export1["/traces/{request_id}"]
    Trace --> Export2["/traces/{request_id}/otel"]

    ToolMiddleware["工具中间件"] --> EventBus["EventBus"]
    EventBus --> Events["tool_start / tool_end / approval_required"]
    Events --> SSE["/chat/stream SSE 外观"]

    Runtime["运行过程"] --> Metrics["MetricsRegistry"]
    Metrics --> Prom["/metrics Prometheus text"]
    Metrics --> Snap["/metrics/snapshot JSON"]
```

可观测性分层：

- **Trace**：定位单次请求走过哪些步骤。
- **Diagnostic event**：解释 Harness 为什么拦截、审批、重试、拒答或完成。
- **Metrics**：观察整体请求量、延迟、工具调用、RAG 评分、token 等趋势。
- **EventBus**：给 SSE 和调试界面输出工具事件、审批事件。

---

## 18. 缓存、重试、超时、熔断架构

```mermaid
flowchart LR
    ToolCall["工具调用"] --> Policy["allowlist + ToolPolicy"]
    Policy --> Cache["ToolCallCache\ntool + args hash"]
    Cache -->|命中| Return["直接返回缓存 ToolMessage"]
    Cache -->|未命中| Retry["RetryPolicy"]
    Retry --> Timeout["timeout_seconds"]
    Timeout --> Breaker["CircuitBreaker"]
    Breaker -->|CLOSED/HALF_OPEN| RealCall["真实工具调用"]
    Breaker -->|OPEN| Fallback["短路返回兜底 ToolMessage"]

    RAGQuery["RAG Query"] --> Semantic["SemanticCache\nembedding 近似命中"]
    Semantic -->|命中| CachedAnswer["返回缓存答案"]
    Semantic -->|未命中| RAGFlow["正常 RAG 检索生成"]
```

说明：

- 工具调用先过策略，再考虑缓存、重试、超时、熔断。
- `ToolCallCache` 只缓存成功的 `ToolMessage`，避免缓存有副作用的 `Command`。
- `CircuitBreaker` 实现 `CLOSED -> OPEN -> HALF_OPEN -> CLOSED` 三态保护。
- RAG 可启用 `SemanticCache`，相似问题命中后跳过检索和生成。

---

## 19. 评测与质量门禁架构

```mermaid
flowchart TB
    Golden["evals/agent_golden.jsonl"] --> Eval["scripts/evaluate_agent.py"]
    Eval --> Mode["mode: agent / harness"]
    Eval --> RunCases["逐条执行 case"]
    RunCases --> TraceTools["从 trace 统计 actual_tools"]
    RunCases --> Keywords["关键词命中率"]
    RunCases --> Reject["安全拒绝率"]
    RunCases --> Latency["P50 / P95 延迟"]
    RunCases --> Cost["平均成本"]

    Latency --> Gate["EvalGate"]
    Cost --> Gate
    TraceTools --> Gate
    Keywords --> Gate
    Reject --> Gate

    Gate --> Thresholds["pass_rate / tool_recall / keyword_recall / p95 / avg_cost"]
    Gate --> Breakdown["rag / tool / report / safety bucket 失败分布"]
    Gate --> Report["storage/agent_eval_report.json"]
```

常用命令：

```powershell
python -m pytest tests -q
python scripts/evaluate_rag.py
python scripts/evaluate_agent.py --dry-run
python scripts/evaluate_agent.py --mode harness --gate --report storage/agent_eval_report.json
python scripts/benchmark_api.py --url http://127.0.0.1:8000/chat --api-key dev-api-key
```

`EvalGate` 不只看整体 pass_rate，还检查 tool_recall、keyword_recall、P95 延迟和平均成本，并按 bucket 拆分失败原因。

---

## 20. 典型请求链路

### 20.1 普通知识库问答

```mermaid
flowchart LR
    User["主刷缠绕毛发怎么办？"] --> API["/chat"]
    API --> Auth["AuthContext + RateLimit + Safety"]
    Auth --> Runner["AgentRunner"]
    Runner --> Backend["ReactAgentBackend"]
    Backend --> Agent["ReactAgent"]
    Agent --> Tool["rag_summarize"]
    Tool --> RAG["Chroma 检索 + hybrid_rank"]
    RAG --> Answer["生成带引用答案"]
    Answer --> Verifier["AnswerVerifier"]
    Verifier --> Artifact["final-answer artifact"]
    Artifact --> Resp["返回 completed"]
```

### 20.2 普通用户生成使用报告

```mermaid
flowchart LR
    User["生成本月使用记录报告"] --> API["/harness/run 或 /chat"]
    API --> Runner["AgentRunner"]
    Runner --> Policy["ToolPolicy"]
    Policy --> Pending["pending_approval"]
    Pending --> ApprovalID["返回 approval_id"]
    ApprovalID --> Operator["operator/admin approve"]
    Operator --> Rerun["携带 approval_id 再次请求"]
    Rerun --> Backend["ReactAgentBackend"]
    Backend --> Agent["ReactAgent + tools"]
    Agent --> Report["报告答案"]
    Report --> Verifier["AnswerVerifier"]
    Verifier --> Artifact["final-answer artifact"]
```

### 20.3 MCP 调用敏感工具

```mermaid
flowchart LR
    Client["MCP Client"] --> MCP["tools/call fetch_external_data"]
    MCP --> Auth["AuthContext"]
    Auth --> Policy["ToolPolicy"]
    Policy -->|need_approval| Store["ApprovalStore pending"]
    Store --> Pending["返回 pending_approval + approval_id"]
    Policy -->|allow 或 approved| Handler["fetch_external_data handler"]
    Handler --> Result["返回工具结果"]
```

---

## 21. 目录结构与职责

| 目录 / 文件 | 职责 |
|---|---|
| `app.py` | Streamlit 聊天演示入口 |
| `api/server.py` | FastAPI 服务入口，集成 auth、harness、approval、artifact、MCP、trace、metrics、judge |
| `agent/react_agent.py` | ReAct Agent 执行面，封装 LangChain create_agent、tools、middleware、memory |
| `agent/runner.py` | Harness 控制核心：AgentRunner、AgentTask、AgentBackendResult、ReactAgentBackend |
| `agent/state.py` | AgentState、Budget、StepRecord、Observation、ToolCallRecord、ArtifactRef |
| `agent/policies.py` | ToolPolicy、PlanValidator、Replanner |
| `agent/verifier.py` | AnswerVerifier 与 VerifyResult |
| `agent/tools/` | LangChain 工具、工具注册表、中间件、重试策略 |
| `agent/workflows/` | 显式业务工作流，目前核心是个人使用报告生成 |
| `rag/` | Chroma 向量库、RAG 检索总结、引用、评测辅助、EvalGate |
| `model/` | 模型工厂、Provider 抽象、多模型路由 |
| `services/approval_store.py` | SQLite 审批状态存储 |
| `services/artifact_store.py` | SQLite 运行产物存储 |
| `services/` | 数据服务、SQLite 持久化、缓存、限流、任务队列、熔断器 |
| `safety/auth.py` | 可信 AuthContext、角色解析、审批角色判断 |
| `safety/security.py` | Prompt 注入检测、RAG 注入检测、工具参数校验、脱敏、审批上下文 |
| `observability/` | trace、diagnostic event、metrics、事件总线、请求上下文 |
| `mcp_adapter/` | MCP JSON-RPC 适配层 |
| `config/` | Agent、RAG、Chroma、Prompt 配置 |
| `data/` | 知识库文件和外部使用记录数据 |
| `docs/` | demo 说明、面试讲稿、Harness 讲稿和架构说明 |
| `tests/` | 单元测试、Prompt 回归、安全、MCP、RAG、Harness 测试 |
| `evals/` | RAG / Agent golden set 评测数据 |

---

## 22. 启动与部署架构

```mermaid
flowchart LR
    Env[".env\nDASHSCOPE_API_KEY / API Key / DB Path"] --> Install["pip install -e .[dev]"]
    Install --> LoadKB["python -m rag.vector_store"]
    LoadKB --> FastAPI["uvicorn api.server:app"]
    LoadKB --> Streamlit["streamlit run app.py"]
    FastAPI --> Docker["Dockerfile\npython:3.10-slim"]
```

Docker 部署：

```powershell
docker build -t sweeper-agent .
docker run --env-file .env -p 8000:8000 sweeper-agent
```

---

## 23. 核心亮点

第一，**Agent 不是黑盒执行**。新版主链路通过 `AgentRunner` 显式管理状态、预算、审批、验证、artifact 和 diagnostic trace。

第二，**敏感工具有真实治理闭环**。`fetch_external_data` 这类 read_sensitive 工具不再只靠 Prompt 约束，而是由 `ToolPolicy`、`ApprovalStore`、operator/admin 审批和工具中间件共同控制。

第三，**执行面和控制面解耦**。`ReactAgent` 继续负责 ReAct 推理和工具调用，`ReactAgentBackend` 作为适配层接入 Harness，后续可替换为其他 Agent 框架。

第四，**RAG 有完整工程链路**。包括文档加载、MD5 去重、chunk 切分、metadata、Chroma 存储、检索、hybrid rank、RAG 注入过滤、引用来源、evidence trace 和评测集。

第五，**工具体系生产化**。工具有 manifest、scope、risk_level、side_effect、requires_approval、timeout_seconds、allowlist、ToolPolicy、缓存、重试、超时、熔断和 trace。

第六，**质量可以被量化**。Agent 评测支持 golden set、工具命中率、关键词命中率、安全拒绝率、P95 延迟、平均成本和失败 bucket。

---

## 24. 当前边界与后续演进

- 当前存储使用 SQLite，适合本地演示和轻量部署；生产环境建议迁移到 Postgres / MySQL，并把缓存、限流和审批短期状态接 Redis。
- `/chat/stream` 当前是 harness 最终结果的 SSE 外观，不是逐 token 透传旧 ReAct stream；如果需要更强实时体验，可把 backend chunk 与 harness 状态事件统一流式化。
- 审批目前以工具级和场景级控制为主；后续可下沉到真实 user_id / month / 参数级审批，并增加审批过期、一次性消费和审批范围校验。
- Budget 已接入 steps、tool calls、tokens、cost 的结构；真实 token / cost 依赖模型 provider 的 usage 质量，可进一步统一在 ModelRouter 层采集。
- ToolPolicy 当前是代码规则；生产环境可升级为 RBAC / ABAC 配置中心，按租户动态下发策略。
- Trace 当前是内存 + SQLite payload；生产环境可接 OpenTelemetry Collector、Jaeger、Tempo 或日志平台。
