# Evaluation Analysis Changelog

## agent-eval-analysis-v1 — 2026-08-12

- 新增完整阶段四 baseline/candidate 报告的同 Case 配对比较，明确与 evaluator-health baseline 分离。
- 新增配对 bootstrap 置信区间、exact binary paired test 与 score sign test。
- 新增协议、数据集、案例集、Judge、Prompt、配置和审批哈希可比性检查。
- 新增 scene/category/risk/capability 切片、通过状态迁移、安全/veto/P0/L3 零容忍门禁。
- 新增只基于 dev/regression 的疑似根因、人工待审迭代建议和 regression 候选。
- promotion 仅接受冻结 test，且所有转移与安全结论只输出聚合计数，不输出逐案身份、Bad Case 或调参建议；最高结论为人工审批资格。
- 输入重新校验 Rubric 分数、veto、overall/pass 语义，并要求 machine-eval 配置与 Rubric 指纹匹配服务端认可版本。
- API 新增 baseline 审批申请入口，租户隔离的服务端审批记录绑定完整报告 SHA-256；客户端自报审批不能用于 promotion。
- 新增 gated slice 综合分门禁，避免全局均值掩盖局部质量退化。
- 新增安全 Markdown 复盘、不可变 SQLite 历史、租户隔离、趋势视图、CLI 和 API。

当前没有创建获批的 Agent promotion baseline，也没有把合成测试结论冒充真实模型提升。
