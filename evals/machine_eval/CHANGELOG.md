# Machine Evaluation Changelog

## agent-machine-eval-v1 — 2026-08-12

- 新增与 `agent-rubric-v1` 对齐的七维、0–3 分结构化 Judge。
- 新增状态、工具选择、参数、事实、引用和 artifact 确定性评分器。
- 新增只降不升的 Hybrid 分数修正规则，并逐条记录 override 依据。
- 新增阶段三最终人工标签对齐：pass F1、agreement、维度 MAE/Kappa、安全和 veto 漏判。
- 新增场景、类别、风险等级和能力标签误差切片。
- 新增 Judge 覆盖率、固定身份、人工样本量、人工批次状态和批准基线生产门禁。
- Judge 错误不再使用中性分参与聚合；每个案例最多重试两次。
- 禁用事实检查区分安全拒绝语境中的提及与实际泄露。
- 为阶段五实验分析透传 family/model lineage、逐案例延迟、成本模式与 token，并补充案例集指纹。
- evaluator-health baseline 在关键人机校准指标缺失时明确标记为 inconclusive，不再返回误导性的 passed。

当前没有创建批准的生产 baseline。首个批准基线必须来自真实关闭、全部完成仲裁的阶段三批次。
