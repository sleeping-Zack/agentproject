# 面向智能硬件客服场景的可治理 Agent 平台

这是一个面向扫地 / 扫拖机器人的 **RAG + 多工具 Agent + Harness 控制层** 项目。项目不仅覆盖知识库问答、天气 / 环境适配、用户设备使用记录查询、型号级故障码与产品规格查询、个性化报告生成和售后工单创建，也把 Agent 应用生产化中常见的控制问题纳入架构：统一状态、预算停止、动态工具策略、真实人工审批、答案验证、artifact 留存、诊断 trace、评测门禁和服务化交付。

一句话概括：

> `ReactAgent` 负责 ReAct 推理和工具调用，`AgentRunner` 负责请求级控制；工具、RAG、MCP、审批、artifact、trace、metrics 都围绕 Harness 形成可治理的 Agent 运行框架。

---

## 1. 功能概览

- **RAG 知识库**：从 `data/` 中的 PDF / TXT 构建 Chroma 向量库，以真实 Dense 分数和中文 BM25 双路召回，经 RRF 融合及可选 Cross-Encoder 精排后生成 evidence 与引用。
- **多工具 Agent**：支持知识库检索、天气、用户上下文、使用记录、型号级故障码、产品规格、报告上下文和售后工单创建。
- **Harness 控制层**：统一 `AgentRunner` / `AgentState`，支持预算停止、动态工具策略、真实审批、答案验证、artifact 存储和诊断 trace。
- **动态工具治理**：`ToolRegistry` 管工具元数据，`ToolPolicy` 从版本化 YAML 加载 tenant / role / scene / tool / args 规则，输出可审计决策；默认配置已包含 tenant A/B 的功能权限差异。
- **真实 HITL 审批**：受保护的数据访问和售后工单写入会进入 `SQLiteApprovalStore`；`create_support_ticket` 仅在用户明确请求且人工审批通过后执行。
- **答案质量闸门**：`AnswerVerifier` 依次执行结构校验、Claim-Evidence 对齐与危险结论检测，仅在高风险或低置信时选择性调用带超时、显式错误和 fail-closed 语义的 `LLMJudge`；直接 RAG 和 AgentRunner 共用该闸门。
- **产物留存**：`SQLiteArtifactStore` 按 request_id 保存 final answer、verification failure、evidence、tool results 等运行产物。
- **服务化入口**：FastAPI 暴露 `/chat`、`/chat/stream`、`/harness/run`、审批、artifact、MCP、trace、metrics、judge 等接口。
- **MCP 工具服务**：支持 JSON-RPC `initialize`、`tools/list`、`tools/call`；MCP 工具调用同样经过 ToolPolicy 和审批存储。
- **可观测性**：包含 request/tool/model trace、diagnostic event、OpenTelemetry 风格 span、Prometheus 指标，以及带序号、心跳、背压和断线重放的实时 SSE 事件流。
- **评测门禁**：PR 运行 30 条冻结真实检索排名、12 条生成 grounding 正反例和 62 条离线 Agent golden，校验固定阈值及相对基线退化；真实 Rerank、生成与 Judge 评测由独立工作流定期执行。

> 数据边界：内置的 `DemoBot S10 / S20 / X30` 规格与故障码目录是 `demo-v1` 确定性演示数据，不代表任何真实品牌或厂商官方资料。未收录型号或故障码时，Agent 应追问或说明边界，不得猜测和跨型号套用。

---

## 2. 环境与启动

推荐 Python 3.10.x，本项目使用 Python 3.10.11 验证。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,production]"
Copy-Item .env.example .env
```

在 `.env` 中配置 `DASHSCOPE_API_KEY`。

默认的单次执行预算在本地环境和 Docker Compose 中保持一致：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `AGENT_MAX_RUN_TOKENS` | `32000` | 单次运行的 Token 总预算 |
| `AGENT_MAX_MODEL_OUTPUT_TOKENS` | `2000` | 单次模型调用的最大输出 Token |
| `AGENT_MODEL_MAX_RETRIES` | `2` | 模型供应商调用失败后的最大内部重试次数 |
| `AGENT_MAX_TOOL_CALLS` | `8` | 单次运行允许的逻辑工具调用数 |
| `AGENT_MAX_STEPS` | `8` | 单次运行允许的执行步骤数 |
| `AGENT_MAX_COST` | `1.0` | 单次运行允许的费用上限 |
| `AGENT_MAX_VERIFICATION_RETRIES` | `1` | 回答校验失败后的定向修复次数 |
| `AGENT_MIN_REPAIR_TOKENS` | `4500` | 启动定向修复前至少保留的 Token |
| `AGENT_RAG_MAX_OUTPUT_TOKENS` | `1600` | 单次 RAG 生成的最大输出 Token |
| `AGENT_MAX_REACT_RECURSION` | `12` | ReAct 图的最大递归步数，防止无界循环 |
| `AGENT_MEMORY_EXTRACTION_MAX_TOKENS` | `900` | 单次长期记忆提取的最大输出 Token |
| `AGENT_SUMMARY_MAX_TOKENS` | `500` | 单次会话摘要的最大输出 Token |

修改这些值后需重启服务。工具内部对同一逻辑调用的物理重试不应重复占用逻辑工具额度。

加载知识库：

```powershell
python -m rag.vector_store
```

本地开发统一使用固定端口（FastAPI `8000`、Streamlit `8501`）。启动或修改代码后重启：

```powershell
.\scripts\dev.ps1 restart
```

脚本会停止本项目已有的前后端进程，在原端口启动当前代码并完成健康检查。也可单独启停或查看状态：

```powershell
.\scripts\dev.ps1 start
.\scripts\dev.ps1 stop
.\scripts\dev.ps1 status
```

Streamlit 不再直接实例化 Agent；聊天、长期记忆、Planner、审批、artifact、trace
和 metrics 全部通过 FastAPI。生产身份来自服务端绑定的 API Key principal 或经过
HS256 校验的 JWT claims；侧边栏中的 Tenant/User 只能与认证身份一致，不能用于冒充身份。

也可以一次启动完整后端与前端：

```powershell
docker compose up --build
```

随后访问 `http://127.0.0.1:8501`。

启动 MCP stdio server：

```powershell
python mcp_server.py
```

---

## 3. 常用接口

| 接口 | 作用 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /auth/me` | 校验当前认证身份及 tenant/user/role 绑定 |
| `GET /tools/manifest` | 导出工具 manifest |
| `GET /sessions` | 按 tenant/user 查询可恢复的历史会话 |
| `GET /sessions/{session_id}/messages` | 恢复指定用户会话的短期消息 |
| `GET /memory` | 查询当前 tenant/user 的长期事实及衰减分数 |
| `GET /memory/events` | 查询当前用户的情景记忆 |
| `GET /memory/summaries` | 查询当前用户的会话压缩摘要 |
| `GET /memory/procedures` | 查询租户可用的已认证程序记忆 |
| `POST /memory` | 显式新增或版本化更正长期记忆 |
| `DELETE /memory` | 遗忘指定记忆或全部用户记忆 |
| `POST /memory/{memory_id}/review` | 接受或拒绝自动冲突形成的待确认记忆 |
| `POST /chat` | 兼容聊天入口，内部走 Harness |
| `POST /chat/stream` | 实时 SSE 事件流，支持 token、工具、审批、验证、artifact、心跳与重连重放 |
| `POST /harness/run` | 推荐生产入口，返回 status / approval_id / verifier / artifacts |
| `GET /approvals` | 查询当前租户审批队列，仅 operator / admin |
| `GET /approvals/{approval_id}` | 查询审批记录 |
| `POST /approvals/{approval_id}/approve` | 审批通过，仅 operator / admin |
| `POST /approvals/{approval_id}/deny` | 审批拒绝，仅 operator / admin |
| `POST /human-eval/batches` | 创建双人盲评批次，仅 operator / admin |
| `GET /human-eval/batches/{batch_id}/tasks/next` | 评审员领取本人下一项盲评任务 |
| `POST /human-eval/tasks/{assignment_id}/submit` | 按 Rubric 提交本人评分或修订版 |
| `GET /human-eval/batches/{batch_id}/report` | 一致性、进度与最终人评质量报告，仅 operator / admin |
| `GET /human-eval/batches/{batch_id}/export` | 导出匿名双评分与仲裁后标签，仅 operator / admin |
| `POST /machine-eval/runs` | 运行确定性 + 七维 Judge 机评和人工校准，仅 operator / admin |
| `POST /evaluation-analysis/compare` | 对 baseline/candidate 完整报告做配对实验分析，仅 operator / admin |
| `GET /evaluation-analysis/reports` | 查询当前租户的不可变实验报告摘要 |
| `GET /evaluation-analysis/experiments/{experiment_id}/trend` | 查询实验趋势、连续退化与安全告警 |
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
- `/chat/stream` 统一输出 `AgentEvent`；客户端可携带同一 `request_id` 与 `Last-Event-ID` 恢复遗漏事件，跨租户、跨会话或不同 query 复用 request_id 会被拒绝。
- `/mcp tools/call` 由 `MCPToolServer` 执行 ToolPolicy；调用 `fetch_external_data`、`create_support_ticket` 等受控工具时会返回 `pending_approval` 和 `approval_id`，审批通过且参数匹配后才执行。
- `tenant_id / user_id / role` 由 API Key 或 JWT 的服务端 claims 决定。前端不发送可伪造的身份请求头，也不提供客户端角色提权。
- `principal_id` 是登录主体，`data_user_id` 是该主体拥有的使用记录标识。本人单月只读报告免审；跨用户报告进入审批，并把申请人、目标用户和月份绑定到审批记录。
- `/chat`、`/chat/stream` 和 `/harness/run` 都接受 `approval_id`。批准后必须由原申请身份以相同参数继续请求；跨身份或改参数复用审批会被拒绝。
- 本地开发可分别配置 `AGENT_OPERATOR_API_KEY` 和 `AGENT_ADMIN_API_KEY`；生产应使用 `AGENT_API_PRINCIPALS_JSON` 或 JWT/外部身份提供方。

长期记忆采用“候选—校验—版本化—使用”四段式链路：

- 高精度规则只作为低延迟兜底；默认启用结构化模型抽取，覆盖开放表达的用户画像、偏好和长期行为要求。
- 模型不能直接写库。候选必须带稳定语义槽位、作用域、长期有效性、置信度和可在原文定位的证据，再经过敏感数据与 schema 校验。
- 显式更正会形成新版本；自动抽取与现有事实冲突时进入 `pending_confirmation`，不会静默覆盖。
- 模糊遗忘请求拒绝猜测；全局用户策略和偏好每轮注入用户级约束上下文，当前请求及系统/安全规则始终优先。
- `AGENT_MEMORY_MODEL_EXTRACTION_ENABLED=false` 可关闭模型抽取，系统仍保留保守规则路径，但开放表达的召回率会下降。

---

## 4. 最新总体架构图

```mermaid
flowchart TB
    User["用户 / 前端 / 外部系统"] --> Streamlit["Streamlit UI\napp.py"]
    User --> FastAPI["FastAPI 服务\napi/server.py"]
    User --> MCPClient["MCP Client"]

    Streamlit -->|HTTP / SSE| FastAPI

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
    Memory --> LongTerm["LongTermMemoryService\n情景 / 语义 / 程序记忆"]
    LongTerm --> MemoryDB["SQLite / Postgres\n权威记忆存储"]
    LongTerm --> MemoryVector["可选 Chroma\n候选索引"]

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
    Req["HTTP 请求"] --> Auth["绑定 principal 的 API Key\n或 HS256 JWT"]
    Auth --> AuthCtx["resolve_auth_context"]
    AuthCtx --> Tenant["tenant_id\n服务端 claims"]
    AuthCtx --> Role["user_role\n服务端 claims"]
    AuthCtx --> Principal["principal_id\nsub / user_id"]
    Req -. "Header/body 仅作一致性校验" .-> AuthCtx
    Role --> Validate["VALID_ROLES\nuser / operator / admin"]
    Validate --> Rate["RateLimiter"]
    Rate --> Safe["assert_safe_user_input"]
    Safe --> Task["AgentTask"]
```

说明：

- 默认 `principal_api_key` 模式把 API Key 映射到服务端 tenant/user/role。
- `jwt` 模式校验 HS256 签名、`exp`、可选 issuer/audience，并从 claims 读取身份。
- `X-Tenant-ID`、`X-Principal-ID`、`X-User-Role` 仅用于与认证身份做一致性校验。
- `legacy_headers` 只供旧测试或短期迁移，生产环境不得启用。
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
    State --> BudgetCheck["BudgetManager preflight + reserve"]

    BudgetCheck -->|超预算| Blocked["status=blocked"]
    BudgetCheck -->|未超预算| Sensitive["handle_sensitive_report_data"]

    Sensitive --> Policy["ToolPolicy.decide"]
    Policy -->|deny| Rejected["status=rejected"]
    Policy -->|need_approval| Pending["SQLiteApprovalStore\nstatus=pending"]
    Policy -->|allow 或 approved| Backend["ReactAgentBackend"]

    Pending --> ReturnPending["返回 pending_approval\napproval_id"]

    Backend --> Usage["抽取 evidence / tool_results / usage"]
    Usage --> BudgetRecord["commit reservation\nactual tokens / cost / tool calls"]
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
| `create_support_ticket` | `support_ticket:write` | `high` | `write` | `true` |

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
- 售后工单只有在用户明确要求创建且审批通过后才会写入；审批记录绑定原始型号、问题类型、描述和可选故障码。

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
    Middleware --> Event["EventBus\ntool_started/tool_completed/tool_failed"]

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
    Splitter --> Meta["metadata\nsource / title / section / page / chunk_version / doc_id"]
    Meta --> Chroma["Chroma 向量库\nstorage/chroma"]

    Query["用户问题"] --> Dense["Dense Retrieval\n真实 relevance score"]
    Query --> BM25["BM25 Retrieval\njieba 中文分词"]
    Chroma --> Dense
    Chroma --> BM25
    Dense --> RRF["Reciprocal Rank Fusion"]
    BM25 --> RRF
    RRF --> Router["Query Router\n型号/错误码/数值约束旁路"]
    Router --> Structured["结构化 passage\nmetadata + chunk"]
    Structured --> Rerank["可选 Cross-Encoder Rerank"]
    Rerank --> RankFusion["Hybrid rank 90%\nRerank rank 10%"]
    Router -->|旁路或失败| Safety["RAG 注入过滤"]
    RankFusion --> Safety
    Safety --> Prompt["RAG Prompt\ninput + context"]
    Prompt --> LLM["Chat Model"]
    LLM --> Answer["答案 + 引用来源"]
    Safety --> Evidence["trace rag_evidence\n供 verifier 使用"]
```

RAG 问答由 `RagSummarizeService` 实现：Dense 与 BM25 分别产生带真实分数和排名的 `RetrievalCandidate`，RRF 在不混用分值尺度的前提下融合候选。可选 reranker 使用文档标题、产品型号、章节、版本、页码和正文组成结构化输入；`shadow` 模式只记录 Cross-Encoder 排名，`weighted_rrf` 使用人工 dev 选出的 Hybrid 90% 与 BGE 10%。型号、错误码、数值约束查询以及模型失败会保留 Hybrid 顺序。

旧“完全替换排序”实验中，30 条冻结 test 的 Recall@5 从 Hybrid `0.9333` 降到 `0.7389`。本轮 25 条、858 个候选均完成审核；90/10 融合在 dev 上把 Recall@5 从 `0.2739` 提到 `0.2755`、nDCG@5 从 `0.6037` 提到 `0.6066`，没有逐条退化。参数冻结后，test 的 Recall@5 / MRR / nDCG@5 为 `0.9333 / 0.9333 / 0.9035`，与 Hybrid 完全持平。

项目同时提供真实 FastAPI BGE 服务、远程客户端、超时/断路器/Hybrid 回退、dev-only 调参和三组隔离 chunk 索引。当前 CPU 服务平均约 `10s`、P95 约 `12s`，未通过 `1s` 发布门禁，因此生产仍默认关闭，实验默认 `shadow`。完整结果、启动方式和门禁见 [`docs/rerank_engineering.md`](docs/rerank_engineering.md)。

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.reranker_server:app --host 127.0.0.1 --port 8090
```

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

    SQLite --> SessionTable["session_messages\ntenant_id + user_id + session_id + role + content"]
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

长期记忆的关键约束：

- 显式“记住/更正/忘记”在模型生成回答前同步执行，回复中的成功承诺来自真实写入结果。
- 用户行为要求使用结构化 `user_policy`；例如回答前缀由输出策略执行器确定性应用。
- 普通自动提取支持白名单内的多事实，但不会绕过敏感信息检测或覆盖已确认冲突。
- 召回同时要求最低相关度、最低总分和非陈旧状态；无关记忆返回空集合。
- 摘要主键为 `(tenant_id, user_id, session_id)`，并保存来源消息标识及摘要校验摘要。
- 软衰减把低于阈值的事实标记为 `stale`，物理删除仍由独立 retention/forget 流程处理。

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

    Runner["AgentRunner"] --> EventBus["EventBus\n有界队列 + replay buffer"]
    ToolMiddleware["工具中间件"] --> EventBus
    ReactStream["模型 token stream"] --> EventBus
    EventBus --> Events["run / token / tool / approval / verifier / artifact / heartbeat"]
    Events --> SSE["/chat/stream\nid + event + data"]

    Runtime["运行过程"] --> Metrics["MetricsRegistry"]
    Metrics --> Prom["/metrics Prometheus text"]
    Metrics --> Snap["/metrics/snapshot JSON"]
```

可观测性分层：

- **Trace**：定位单次请求走过哪些步骤。
- **Diagnostic event**：解释 Harness 为什么拦截、审批、重试、拒答或完成。
- **Metrics**：观察整体请求量、延迟、工具调用、RAG 评分、token 等趋势。
- **EventBus**：为每个 request 维护严格递增序号、有界 live queue 和短期 replay buffer；慢消费者触发背压取消，断线客户端用 `Last-Event-ID` 重放。

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
    Semantic -->|命中| CachedAnswer["恢复完整 RagResult\nanswer + evidence + citations"]
    Semantic -->|未命中| RAGFlow["正常 RAG 检索生成"]
```

说明：

- 工具调用先过策略，再考虑缓存、重试、超时、熔断。
- `ToolCallCache` 只缓存成功的 `ToolMessage`，避免缓存有副作用的 `Command`。
- `CircuitBreaker` 实现 `CLOSED -> OPEN -> HALF_OPEN -> CLOSED` 三态保护。
- RAG 可启用 `SemanticCache`，缓存完整 `RagResult`；key 隔离 tenant、知识库、语料、prompt、检索和模型版本，避免缓存命中丢失 evidence/citation 或知识库更新后返回旧答案。

---

## 19. 评测与质量门禁架构

统一的人评维度、评分锚点、一票否决、指标口径和标注流程见
[`docs/evaluation_spec.md`](docs/evaluation_spec.md) 与
[`docs/annotation_guideline.md`](docs/annotation_guideline.md)；机器可读定义位于
`config/evaluation_rubric.yml` 和 `config/evaluation_metrics.yml`。

阶段二构建的 175 条分层 Agent 评测规格、split 防泄漏规则与复现命令见
[`docs/evaluation_dataset_v1.md`](docs/evaluation_dataset_v1.md)。该数据集当前仍是
`candidate_pending_human_review`，完成独立人工复核前不作为生产 Golden。

阶段三的双人盲评、Rubric 提交校验、自动分歧检测、质检退回、版本化修订、第三方仲裁、
Kappa/质量报告、匿名标签导出和审计流程见
[`docs/human_evaluation_workflow.md`](docs/human_evaluation_workflow.md)。实现已通过合成流程测试，
但不会把合成评分冒充真实人工标注；数据集状态只有在独立评审员实际完成审核后才能升级。

阶段四的确定性评分、七维 Rubric Judge、保守 Hybrid 合成、人机 F1/MAE/Kappa 校准、
安全漏判检查、分场景误差切片、版本化基线与生产门禁见
[`docs/machine_evaluation_workflow.md`](docs/machine_evaluation_workflow.md)。未接入关闭且完整的
阶段三人工批次时，机评报告会明确标记为候选，不能通过生产门禁。

阶段五把两份完整阶段四报告按同一 `case_id` 做配对实验分析，严格分离评测器健康度与
候选 Agent 质量，输出 bootstrap 置信区间、exact paired test、退化切片、Bad Case 疑似根因、
迭代建议、待审回归候选以及不可变趋势记录。流程与审批边界见
[`docs/evaluation_analysis_workflow.md`](docs/evaluation_analysis_workflow.md)：`diagnostic` 仅使用
dev/regression 做迭代，`promotion` 仅使用冻结 test 且最高只能给出
`eligible_for_human_approval`，只输出聚合结论，不泄露 test 逐案身份，也不会自动发布或升级基线。
API promotion 的 baseline 先通过 `/evaluation-analysis/baseline-approvals` 创建待审批记录，再绑定当前租户审批存储中的 `baseline_approval_id`；CLI 审批文件仅用于离线复现。

```mermaid
flowchart LR
    B["Baseline 阶段四完整报告"] --> Compare["同 Case 配对分析"]
    C["Candidate 阶段四完整报告"] --> Compare
    Compare --> Health["评测器健康门禁"]
    Compare --> Quality["质量 / 性能 / 安全证据"]
    Quality --> RCA["退化切片与疑似根因"]
    RCA --> Backlog["人工审核的迭代与回归候选"]
    Health --> Decision["blocked / diagnostic_only / keep_baseline / eligible_for_human_approval"]
    Quality --> Decision
    Decision --> Human["发布负责人最终审批"]
```

```mermaid
flowchart TB
    RetrievalGolden["30-case retrieval golden"] --> RetrievalEval["冻结真实排名评测"]
    RetrievalFixture["Dense / Hybrid fixture"] --> RetrievalEval
    RetrievalBaseline["retrieval baseline"] --> RetrievalEval

    GenerationGolden["12-case generation golden"] --> GenerationEval["事实/禁止事实/引用/拒答"]
    GenerationBaseline["generation baseline"] --> GenerationEval
    AgentGolden["62-case offline Agent golden"] --> AgentEval["完整 AgentRunner 离线执行"]
    AgentBaseline["agent baseline"] --> AgentEval

    RetrievalEval --> Gate["固定阈值 + baseline delta"]
    GenerationEval --> Gate
    AgentEval --> Gate
    Gate --> CI["PR blocking CI"]
    Online["真实模型 / 真实向量库"] --> Scheduled["定期与手动报告\n不阻断 PR"]
```

常用命令：

```powershell
python -m pytest tests -q
python -m ruff check .
python -m scripts.evaluate_retrieval --fixture evals/fixtures/retrieval_rankings_v1.json --baseline evals/baselines/retrieval_baseline_v1.json --gate
$env:AGENT_RERANK_STRATEGY="weighted_rrf"; python -m scripts.evaluate_retrieval --enable-reranker --candidate-k 20 --report reports/retrieval-rerank-online.json
python -m scripts.evaluate_generation --baseline evals/baselines/generation_baseline_v1.json --gate
python -m scripts.evaluate_agent --golden evals/agent_offline_golden.jsonl --mode harness --offline --baseline evals/baselines/agent_baseline_v1.json --gate --gate-profile offline_fixture
python -m scripts.analyze_evaluation_experiment --baseline reports/baseline.json --candidate reports/candidate.json --experiment-id exp-routing-v2 --mode promotion --hypothesis "routing-v2 is non-inferior" --change "router-v1 -> router-v2" --baseline-approval reports/baseline-approval.json --output reports/exp-routing-v2.json --markdown reports/exp-routing-v2.md --gate
python -m scripts.benchmark_api --url http://127.0.0.1:8000/chat/stream --profile staging --gate
```

PR 门禁同时检查检索 Recall / Precision / MRR / nDCG，生成软质量与高风险硬约束，以及 Agent 的适用样本分母、风险分层、分桶覆盖、Mock Harness 控制面耗时和相对基线退化。部署环境性能由独立工作流检查并发、成功 QPS、首 Token、完整响应 P95/P99、超时率以及模型/工具阶段耗时。完整阈值和解释见 [`docs/ci_quality_gates.md`](docs/ci_quality_gates.md)。

九项原始 P0 / P1 的实现映射、实测指标和最终验收命令见 [`docs/p0_p1_completion.md`](docs/p0_p1_completion.md)。

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
    Tool --> RAG["Dense + BM25 + RRF + 可选 Rerank"]
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
| `agent/state.py` / `agent/budget.py` | AgentState、统一 BudgetManager、预算预留与提交、StepRecord、Observation、ToolCallRecord |
| `agent/policies.py` / `config/tool_policy.yml` | 版本化、按租户加载且可审计的 ToolPolicy，以及 PlanValidator、Replanner |
| `agent/verifier.py` | AnswerVerifier 与 VerifyResult |
| `agent/tools/` | LangChain 工具、工具注册表、中间件、重试策略 |
| `agent/workflows/` | 显式业务工作流，目前核心是个人使用报告生成 |
| `rag/` | Chroma、Dense/BM25/RRF/Rerank、完整语义缓存、RAG 生成、引用与评测 |
| `model/` | 模型工厂、Provider 抽象、多模型路由 |
| `services/approval_store.py` | SQLite 审批状态存储 |
| `services/artifact_store.py` | SQLite 运行产物存储 |
| `services/` | 数据服务、SQLite 持久化、缓存、限流、任务队列、熔断器 |
| `safety/auth.py` | 可信 AuthContext、角色解析、审批角色判断 |
| `safety/security.py` | Prompt 注入检测、RAG 注入检测、工具参数校验、脱敏、审批上下文 |
| `observability/` | trace、diagnostic event、metrics、事件总线、请求上下文 |
| `mcp_adapter/` | MCP JSON-RPC 适配层 |
| `config/` | Agent、RAG、Chroma、Prompt 配置 |
| `data/` | 知识库文件、外部使用记录，以及 Demo 产品规格和型号级故障码数据 |
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

第四，**RAG 有完整工程链路**。包括文档加载、指纹去重、chunk 与稳定 ID、Chroma Dense、中文 BM25、RRF、可选 Cross-Encoder、注入过滤、完整结果缓存、引用、evidence trace 和真实检索评测。

第五，**工具体系生产化**。工具有 manifest、scope、risk_level、side_effect、requires_approval、timeout_seconds、allowlist、ToolPolicy、缓存、重试、超时、熔断和 trace。

第六，**质量可以被量化且阻断退化**。PR 对 30 条检索、12 条生成 grounding 和 62 条 Agent 案例执行确定性门禁，并同时比较固定阈值与版本化 baseline；线上模型、Cross-Encoder 与 Judge 评测独立运行并产出 artifact。

---

## 24. 多实例部署

多实例演进已经落地，同时保留本地轻量模式：

- 持久化通过 `AGENT_STORAGE_BACKEND` 切换 SQLite / Postgres；会话、trace、长期记忆、审批、产物、人工评测和评测分析都可由 Postgres 共享。
- approval 与 artifact 按 request 维度幂等；审批状态只允许从 `pending` 原子转换一次，避免两个实例同时覆盖审批结果。
- `AGENT_EVENT_BUS_BACKEND=redis` 启用 Redis Streams，跨实例共享生产者归属、严格递增序号、SSE replay、关闭和取消状态。
- 限流和工具调用缓存可分别通过 `AGENT_RATE_LIMIT_BACKEND=redis` 与 `AGENT_CACHE_BACKEND=redis` 共享。
- `AGENT_OTEL_ENABLED=true` 启用 OpenTelemetry SDK、FastAPI 自动埋点与 OTLP HTTP 导出；本地 trace payload 和 `/traces/{request_id}/otel` 保持兼容。

完整生产拓扑可直接启动：

```powershell
docker compose up --build
```

服务地址：API `http://127.0.0.1:8000`，Jaeger UI `http://127.0.0.1:16686`。Compose 同时启动 Postgres、Redis、OpenTelemetry Collector 和 Jaeger，配置见 `docker-compose.yml` 与 `deploy/otel-collector.yaml`。

从本地 SQLite 切换时，先启动依赖，再执行可重复运行的迁移命令。该命令会逐表校验，且不会删除 SQLite 源文件：

```powershell
docker compose up -d postgres redis
python -m scripts.migrate_sqlite_to_postgres
```

本地 Docker PostgreSQL 默认映射到 `127.0.0.1:55432`，避免与系统 PostgreSQL 的标准端口冲突；容器网络内仍使用 `postgres:5432`。

仅运行分布式后端集成测试时：

```powershell
docker compose up -d postgres redis
$env:AGENT_TEST_POSTGRES_URL="postgresql://agent:agent@127.0.0.1:55432/agent"
$env:AGENT_TEST_REDIS_URL="redis://127.0.0.1:6379/0"
pytest tests/test_distributed_backends.py -q
```

仍需根据实际业务继续下沉的边界包括审批过期与一次性消费、集中式 RBAC/ABAC、数据库备份恢复和生产级 Jaeger/Tempo 持久化存储。当前 Compose 中的 Jaeger 使用内存存储，只适合本地验证链路。
