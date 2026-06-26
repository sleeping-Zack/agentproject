# Agent Harness 控制层面试讲稿

## 一句话介绍

这次升级把项目从“Agent 能调用工具”推进到“有统一控制层的 Agent 应用”。核心是新增 `AgentRunner`，它不替代 LangChain ReAct，而是在外层负责状态、预算、策略、审批、验证、artifact 和诊断 trace。

## 为什么要补 Harness

原来的主链路是 `ReactAgent.execute_stream()`，适合 Demo，但生产系统还需要回答几个问题：一次请求现在走到哪一步、最多能调用几次工具、敏感工具是否经过审批、最终回答有没有证据支撑、失败时能不能暂停/拒答/追踪。Harness 层就是把这些控制职责从 prompt 和散落模块里收拢出来。

## 十个缺口对应改造

1. 统一 Runner/Controller：新增 `agent/runner.py`，`AgentRunner.run()` 统一创建状态、检查预算、执行策略、处理审批、调用后端、验证回答、保存 artifact。
2. 统一 AgentState：新增 `agent/state.py`，包含 request/session/tenant/user_goal/scene/plan/steps/observations/tool_calls/artifacts/budget/status/final_answer。
3. 真实 HITL：新增 `services/approval_store.py` 和 API `/approvals/{id}/approve|deny`，敏感工具先进入 `pending_approval`，不再用 `confirmed=True` 假装确认。
4. 结果验证闭环：新增 `agent/verifier.py`，把引用/证据检查和 `rag.judge.LLMJudge` 封装成 `AnswerVerifier`，失败时 retry 或拒答。
5. Planner 加保护：新增 `agent/policies.py` 里的 `PlanValidator` 和 `Replanner`，能检查任务类型、依赖和步数预算，失败后生成兜底任务。
6. 动态工具权限：`ToolPolicy` 根据 tenant/user_role/scene/tool/args 返回 allow、deny、need_approval、need_redaction。
7. 工具风险元数据：`ToolSpec` 增加 `risk_level`、`side_effect`、`requires_approval`、`timeout_seconds`，manifest 不再只是名字和 schema。
8. Artifact 管理：新增 `services/artifact_store.py`，把最终回答、验证失败结果、证据和后续评测报告按 request_id 存起来。
9. 诊断级 Trace：`TraceRecorder.record_diagnostic_event()` 记录 step_id、tool、args_hash、tokens、cost、evidence_ids、verifier、retry、prompt_version、model_name、failure_reason。
10. 模型路由接入主链路：`model/factory.py` 的 `ChatModelFactory` 改为通过 `model_router.invoke()` 获取模型，默认模型也走 provider/router 抽象。
11. Agent 评测门禁：`rag/eval_gate.py` 和 `scripts/evaluate_agent.py --gate` 支持 pass_rate、tool_recall、keyword_recall、P95 延迟、平均成本、bucket 失败分布。

## 核心链路怎么讲

面试时可以按这条链路讲：

```text
HTTP /harness/run
  -> AgentRunner 创建 AgentState 和 Budget
  -> ToolPolicy 判断是否允许工具
  -> 敏感工具进入 SQLiteApprovalStore pending_approval
  -> 审批通过后继续调用 ReactAgentBackend
  -> AnswerVerifier 校验证据、引用和 judge 分数
  -> SQLiteArtifactStore 保存最终回答或验证失败
  -> TraceRecorder 输出诊断事件
```

这个设计的关键取舍是：我没有重写 LangChain 的 ReAct token loop，而是把可观测、可控、可审批、可验证的职责放在外层 Runner。这样既保留现有功能，又能展示真实工程里的控制面能力。

## 代码讲解

`agent/state.py`：这是统一状态对象。`Budget` 控制最大步数、工具调用、token 和 cost；`AgentState.status` 明确区分 `running`、`pending_approval`、`blocked`、`failed`、`rejected`、`completed`。

`agent/runner.py`：这是 harness 核心。它先判断报告类请求是否会触发 `fetch_external_data`，普通用户会被暂停到审批；admin 或已审批请求才继续执行后端。后端执行完后会进入 `AnswerVerifier`，通过才写 final-answer artifact。

`agent/policies.py`：这是策略层。它不是简单 allowlist，而是把用户角色、场景和工具风险结合起来。例如 `fetch_external_data` 只有 report 场景可用，普通用户需要审批，admin 可以直接读。

`services/approval_store.py`：这是真实 HITL 状态机。审批记录有 pending、approved、denied，包含 request_id、tenant_id、tool_name、args、reason、decided_by 和 decided_at。

`agent/verifier.py`：这是回答质量闸门。对于有 evidence 的回答，要求输出里有引用；如果接入 LLMJudge，还会看 correctness、faithfulness、completeness 的总体分数。

`services/artifact_store.py`：这是产物层。回答、验证失败、报告、检索结果都可以按 request_id 落库，方便复盘一次完整运行。

`rag/eval_gate.py`：这是 CI/评测门禁。它不只看整体通过率，还能按 rag/tool/report/safety bucket 拆失败原因，并限制 P95 延迟和平均成本。

## 面试回答模板

如果面试官问“你的 Agent 怎么证明稳定”，可以回答：

我做了三层验证。第一层是单测，覆盖工具、RAG、工作流、安全、模型路由和 harness 状态机；第二层是 golden set 评测，评估工具命中、关键词命中和 RAG 指标；第三层是 eval gate，把 pass_rate、tool_recall、P95 延迟和失败 bucket 做成 CI 可执行门禁。这样不是靠主观演示，而是能复现地证明系统质量。

如果面试官问“敏感工具怎么处理”，可以回答：

我没有让模型自己决定是否能读敏感数据，而是在 harness 外层做策略裁决。`ToolPolicy` 先根据 role、scene 和工具风险返回 decision；如果是普通用户生成报告，会创建 pending approval 并暂停运行，审批 API approve 后才允许继续。拒绝审批时 runner 会返回 rejected，不会调用工具。

如果面试官问“为什么需要 artifact”，可以回答：

Agent 输出不是只有最终 answer。真实系统还要保留检索证据、工具结果、报告内容、验证失败原因和评测报告。ArtifactStore 把这些都挂到 request_id 下，后续排查、回放和质量分析都能找到依据。

如果面试官问“你怎么做可观测”，可以回答：

普通 trace 只能看工具名和耗时，我又补了 diagnostic event，包含 step_id、状态、工具、参数哈希、证据 ID、verifier 结果、retry 次数、prompt 版本、模型名和失败原因。这样线上问题可以定位到“哪个步骤、哪个策略或哪个验证环节失败”。

## 仍可继续演进

当前 harness 是轻量生产控制层，下一步可以把 SQLite 换成 Postgres，把审批接企业 IM，把 trace 接 OpenTelemetry Collector，把 ToolPolicy 接 RBAC/ABAC 配置中心，把 AnswerVerifier 改成异步多评委投票。
