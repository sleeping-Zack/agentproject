# Agent Quality Dataset Changelog

## agent-quality-v1 — 2026-08-10

- 建立统一的端到端 Agent 案例 Schema、manifest 和确定性 split。
- 发布 175 条候选案例，覆盖六类数据和七个 Rubric 主场景。
- 将 50 条真实 RAG 问题迁移为带来源、工具和风险标签的端到端规格。
- 新增工具、多步骤、安全、异常恢复与对抗边界案例。
- 加入标准化重复、近重复、语义家庭、冻结 test/alias 泄漏和 SHA-256 来源校验。
- 当前状态为 `candidate_pending_human_review`；尚未完成双人盲评，不允许作为生产 Golden。
