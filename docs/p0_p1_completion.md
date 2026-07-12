# P0 / P1 最终验收记录

本记录对应最初提出的 5 项 P0 和 4 项 P1。九项均已完成，并由自动化测试、CI 门禁或真实在线评测覆盖。验收以实现和可复现证据为准，不以功能描述代替测量结果。

| 优先级 | 原问题 | 完成状态 | 核心实现 | 验收证据 |
|---|---|---|---|---|
| P0 | 1. Hybrid RAG 不是真正的混合检索 | 完成 | `rag/retrievers/` 实现真实 Dense 分数、中文 BM25、RRF 和可选 BGE Cross-Encoder；`rag/rag_service.py` 接入统一候选结构 | 30 条真实语料评测：Hybrid Recall@5 `0.9333`、MRR `0.9333`、nDCG@5 `0.9035`，均优于 Dense；见 `evals/reports/retrieval_online_summary_v1.json` |
| P0 | 2. RAG 评测无法证明检索效果 | 完成 | 检索与生成分开评测；冻结真实候选排名、文档 ID golden、版本化 baseline，并计算 Recall / Precision / MRR / nDCG / Hit Rate / 延迟 | `scripts/evaluate_retrieval.py`、`scripts/evaluate_generation.py` 和 CI blocking gate |
| P0 | 3. Agent CI 门禁无效 | 完成 | 62 条离线 golden 真实执行 `AgentRunner`，覆盖路由、参数、审批、租户、预算、安全、artifact、引用、缓存和异常；固定阈值与 baseline delta 双门禁 | `evals/agent_offline_golden.jsonl`、`evals/baselines/agent_baseline_v1.json`、`.github/workflows/ci.yml` |
| P0 | 4. 会话消息重复持久化 | 完成 | 持久化责任收敛到会话提交阶段；按 `tenant_id + session_id + request_id + role` 幂等；按终态决定是否提交最终回答 | `agent/memory.py`、`services/persistence.py`、`tests/test_memory.py`、`tests/test_persistence_tenant.py` |
| P0 | 5. `/chat/stream` 非实时流 | 完成 | 统一事件流支持 token/tool/verification/approval/completion、单调序号、heartbeat、取消和 `Last-Event-ID` 重放 | `agent/runner.py`、`observability/event_bus.py`、`api/server.py`、`tests/test_agent_streaming.py` |
| P1 | 6. AnswerVerifier 只有格式校验 | 完成 | 结构校验、Claim-Evidence 对齐、危险/矛盾规则与选择性 Judge；Judge 超时或异常 fail-closed，高风险和低置信结果才调用语义层 | `agent/answer_schema.py`、`agent/verifier.py`、`rag/judge.py`、`tests/test_answer_verifier_grounding.py` |
| P1 | 7. 语义缓存丢失 Evidence / Citation | 完成 | 缓存完整 `RagResult`；key 隔离 tenant、知识库、语料、prompt、检索和模型版本 | `rag/rag_service.py`、`services/cache.py`、`tests/test_rag_result.py` |
| P1 | 8. ToolPolicy 未按租户动态配置 | 完成 | 外部化版本策略按 tenant / role / scene / tool / args 决策，输出规则 ID、版本、脱敏参数并留审计记录；默认 tenant-a 与 tenant-b 行为不同 | `agent/policies.py`、`config/tool_policy.yml`、`tests/test_policy_engine.py` |
| P1 | 9. Token / Cost 预算事后阻断 | 完成 | 模型和工具调用前原子预留，完成后按真实用量提交或释放；并发不超卖，剩余预算约束最大输出 token | `agent/budget.py`、`agent/tools/middleware.py`、`tests/test_budget_manager.py`、`tests/test_model_budget_middleware.py` |

## 实测结论

- Hybrid 在同一 30 条查询和真实语料上优于 Dense：Recall@5 从 `0.8056` 提升到 `0.9333`，nDCG@5 从 `0.7683` 提升到 `0.9035`。
- BGE reranker 在当前 CPU 环境使 Recall@5 降到 `0.7389`，平均延迟增至 `29352.216 ms`。因此保留可选能力但不默认启用，后续只有在模型或服务运行时通过同一基线后才晋级。
- 12 条真实生成评测全部通过：事实覆盖、拒答准确率、引用有效率和引用覆盖率均为 `1.0`，禁止事实和危险指令命中率均为 `0.0`；Judge 12/12 成功，correctness 与 faithfulness 均为 `5.0/5`。报告同时保留词法粗筛率，Judge 校正后的 unsupported claim rate 为 `0.0`。详见 `evals/reports/generation_online_summary_v1.json`。

## 最终质量门禁

```powershell
python -m pytest tests -q
python -m ruff check .
python scripts/validate_retrieval_manifest.py
python scripts/evaluate_retrieval.py --fixture evals/fixtures/retrieval_rankings_v1.json --baseline evals/baselines/retrieval_baseline_v1.json --gate --gate-strategy hybrid
python scripts/evaluate_generation.py --baseline evals/baselines/generation_baseline_v1.json --gate --max-unsupported-claim-rate 0.05
python scripts/evaluate_agent.py --golden evals/agent_offline_golden.jsonl --mode harness --offline --baseline evals/baselines/agent_baseline_v1.json --gate --min-case-count 60
```
