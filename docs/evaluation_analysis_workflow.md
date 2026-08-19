# 第五阶段：评测分析与模型迭代闭环

本文定义 Agent 评测第五阶段的标准流程：以第四阶段产出的两份机评报告为输入，对候选版本与基线版本做同 Case 配对分析，完成统计判断、切片退化定位、Bad Case 根因归因、回归候选提案和发布建议。

本阶段不修改模型、不直接改数据集，也不替代发布审批。它将“评测结果”转化为可复核的迭代证据。

## 1. 核心原则

1. **同 Case 配对**：基线和候选必须在相同 Case 集合上运行。不能用两个独立样本的均值差冒充版本收益。
2. **先验证评测器，再判断模型**：输入报告版本、Judge 固定状态、人与机一致性和生产门禁不可信时，发布结论必须阻断。
3. **冻结测试集**：promotion 只能使用冻结的 `test`；测试集失败不得自动生成调优建议或新回归样例。
4. **诊断与发布分离**：`dev` / `regression` 用于定位和迭代；它们的结果只能形成 `diagnostic_only`，不能形成发布资格。
5. **安全零容忍**：新增安全失败、新增一票否决或 P0 退化优先于平均分提升。
6. **机器给证据，人做发布决定**：系统最高只能给出 `eligible_for_human_approval`，不会自动上线。

## 2. 两种运行模式

| 模式 | 允许数据 Split | 用途 | 可生成迭代建议 | 可产生发布资格 |
|---|---|---|---|---|
| `diagnostic` | `dev`、`regression` | 快速定位失败、验证修复方向 | 是，但必须人工审核 | 否 |
| `promotion` | 仅冻结 `test` | 候选与基线的正式发布比较 | 否 | 是，但仍需人工审批 |

promotion 运行一旦读取 `test`，只能产出聚合统计、安全/根因代码汇总和发布判断。`case_transitions` 只保留 improved/regressed/unchanged 计数；`safety` 只保留新增或修复的安全失败、veto、P0、L3 聚合计数，禁止任何 `*_case_ids`、`new_vetoes`、`resolved_vetoes` 等逐案字段；`bad_cases`、`recommendations`、`regression_candidates.proposed` 与 `regression_candidates.excluded` 必须全部为空。禁止把测试集问题转写为 Prompt、训练数据、检索语料或回归样例；后续修复应在独立的 dev 数据上复现，再进入下一轮正式测试。

## 3. 输入契约与可比性

输入为第四阶段产生的 baseline/candidate 机评报告。每份输入都记录 SHA-256 和身份信息，以保证复盘时能准确追溯。

正式配对前至少检查：

- Case ID 集合、Case 数量、dataset_version 和 split 一致；
- pipeline、rubric、Judge Prompt 和 Judge ID 版本一致且固定；
- machine-eval 配置与 Rubric 指纹必须等于分析服务端认可的文件指纹，而不只是两侧彼此相等；
- 两份机评报告都通过第四阶段生产门禁；
- 每个 Case 均有最终 hybrid 结论，无未决 Judge 错误；
- promotion 只包含冻结 `test`，diagnostic 不包含 `test`；
- baseline 与 candidate 身份和各自报告哈希都被完整记录。同内容的合法重复实验允许两侧哈希相同，系统不能把“哈希不同”误作可比性的必要条件。

任一硬性检查失败时，`comparability.status` 为 `not_comparable`，最终 `release_decision.status` 必须是 `blocked`。警告信息可以保留，但不得被静默忽略。

API promotion 不接受客户端自报的“已审批”对象。先向 `POST /evaluation-analysis/baseline-approvals` 提交完整 baseline 报告，系统重验报告语义、生产门禁和服务端认可的配置指纹后创建待审批记录；再由 operator/admin 通过通用审批接口作出决定。promotion 只接受当前租户审批存储中的 `baseline_approval_id`，并要求记录用途为 `evaluation_analysis_baseline`、`args.report_sha256` 绑定完整 baseline 报告摘要。CLI 的 `--baseline-approval` 仅适用于离线复现；其文件不是服务端可信审批凭据，不能单独作为生产审计证据。

## 4. 三层门禁

报告将发布资格拆成三层，避免把“模型分数变高”错误等同于“可以上线”。

1. **输入可比性门禁**：检查相同 Case、相同协议、相同评测环境和冻结测试集。
2. **评测器可信度门禁**：检查第四阶段 Judge 标定结果、生产门禁和错误率。
3. **证据充分性门禁**：检查配对样本量、主指标、统计不确定性、安全结果、切片覆盖及性能约束。

三层全部通过后，promotion 报告才可能得到 `eligible_for_human_approval`。

## 5. 统计口径

### 5.1 指标

- 通过率：按同 Case 的 binary pass 结果计算 baseline、candidate 与 delta；
- 综合得分及各 Rubric 维度：按同 Case 分数差计算均值 delta；
- 性能：比较 P95 延迟、平均成本与平均 token；成本仅在两侧数据完整且 `cost_mode` 一致时进入门禁，缺失值不会按 0 处理；
- 安全：统计新增/修复的安全失败与 veto Case，新增项直接触发拒绝或阻断；
- 切片：至少覆盖 scene、category、risk_level 和 capability_tags。

### 5.2 置信区间与显著性

- 连续分数使用 **paired bootstrap** 对 Case 差值重采样，diagnostic 默认 2,000 次、promotion 默认 10,000 次、95% 置信区间，并固定随机种子以便复现；
- binary 通过结果使用配对的 exact test，只在 baseline/candidate 不一致的 Case 上计算；
- 连续维度的方向检验使用 paired sign test，平局不进入有效样本数；
- 小样本切片可以展示，但只有达到配置的 gated slice size 才能独立触发发布门禁；
- `p < 0.05` 不是唯一判断条件。点估计、置信区间、安全红线、业务阈值和样本量必须共同解释。

promotion 至少要求 30 个完整配对案例。预注册主指标是通过率，默认非劣界限为 2 个百分点：只有 95% CI 下界严格大于 `-0.02` 才能认定 non-inferior；CI 下界大于 0 且 exact paired test 的 `p < 0.05` 才能称为 superior。边界相等或区间跨界均为 inconclusive，不能把“未显著”写成“两者相同”。

报告中的区间是“当前离线 Case 分布下的抽样不确定性”，不代表线上流量变化、数据漂移或新工具故障已经被覆盖。

## 6. Case 迁移、Bad Case 与根因

系统按 Case 记录以下迁移：baseline fail → candidate pass、baseline pass → candidate fail、双方通过、双方失败；同时比较维度分、veto、安全结果、确定性检查、性能与成本。

diagnostic 的 Bad Case 报告只输出结构化摘要：`case_id`、优先级、错误类型、证据代码、根因、责任模块和建议动作。promotion 不输出任何逐案 Bad Case。报告渲染器不会输出用户 query、模型原 answer、完整 trace 或任意上游附加字段，防止复盘文档二次泄露敏感内容。

根因聚合用于回答三个问题：

- 哪类错误影响最多 Case；
- 责任模块是谁；
- 修复后应重跑哪些已知 Case 和哪些独立回归候选。

根因是“基于可见证据的工程归因”，不是事实定论。跨模块或无法复现的问题必须交由人工复核。

## 7. 回归候选与数据防污染

仅 diagnostic 模式可从 `dev` / `regression` 失败中产生 `proposed` 回归候选。候选只是待审核提案，不会自动写入数据集。

人工审核至少确认：

- 失败可复现，且不是评测器误判；
- 预期行为、证据和错误类型可操作；
- 新 Case 与原 Case 不只是同义改写；
- 不含个人信息、凭据或内部敏感数据；
- 不来自冻结 test，也没有泄露 test 的特定答案；
- 通过 `proposed → approved → implemented` 生命周期后才进入 regression。

## 8. 发布决策与人工审批边界

`release_decision.status` 只有四种：

| 状态 | 含义 | 下一步 |
|---|---|---|
| `blocked` | 输入不可比、评测器不可信或关键数据缺失 | 修复评测链路后重新运行 |
| `diagnostic_only` | 使用 dev/regression 或证据不足，结论仅供迭代 | 处理建议，完成 promotion 运行 |
| `keep_baseline` | 触发质量、安全、性能或显著切片退化 | 保留基线，修复后重新评测 |
| `eligible_for_human_approval` | promotion 三层门禁通过 | 由发布负责人进行最终审批 |

必须人工审批的操作包括：生产发布、接受风险豁免、调整发布阈值、批准回归候选、修改冻结测试集、使用测试集失败指导模型调优。自动化系统不得把 `eligible_for_human_approval` 解释为已批准或自动 promote。

## 9. CLI 与 API

CLI 入口：

```powershell
python scripts/analyze_evaluation_experiment.py `
  --baseline reports/baseline.json `
  --candidate reports/candidate.json `
  --experiment-id exp-routing-v2 `
  --mode promotion `
  --hypothesis "routing-v2 improves pass rate without safety regression" `
  --change "router-v1 -> router-v2" `
  --output reports/exp-routing-v2.json `
  --markdown reports/exp-routing-v2.md
```

CLI 应使用非零退出码表达被阻断或保留基线，便于接入 CI，但不会执行发布。

API 前缀为 `/evaluation-analysis/*`，用于申请 baseline 审批、提交配对分析、读取不可变报告、列出实验运行和查看趋势。API 必须保持 tenant 隔离；分析请求只接受机评报告结构，不接受任意脚本或外部 URL。具体请求/响应字段以 OpenAPI 和 `evals/evaluation_analysis/report_schema_v1.json` 为准。

主要端点：

| Endpoint | 用途 |
|---|---|
| `POST /evaluation-analysis/compare` | 比较两份完整报告并可持久化结果 |
| `GET /evaluation-analysis/reports` | 按 tenant/experiment 列出不可变摘要 |
| `GET /evaluation-analysis/reports/{report_id}` | 读取单份完整报告 |
| `GET /evaluation-analysis/experiments/{experiment_id}/trend` | 查看连续退化和安全告警趋势 |

阶段五不会在仓库中预置伪造的 baseline/candidate 生产报告，也不会在 CI 中用合成 Judge 结论冒充真实提升。真实阶段四报告、审批哈希和线上成本数据齐备后，可由定期或手动 workflow 调用上述 CLI；当前合成数据只用于契约测试。

## 10. 报告与审计

JSON 报告是机器可读的事实来源，必须符合 `report_schema_v1.json`。Markdown 是面向评审的安全摘要，可由 `evaluation_analysis.reporting.render_markdown` 或 `write_markdown` 生成。

每份报告至少保留：

- analysis_version、配置哈希、输入报告哈希；
- 实验假设、变更、主指标与运行模式；
- 三层门禁、样本量、KPI/CI、切片、安全与性能；
- Case 迁移、结构化 Bad Case、根因、建议与回归候选；
- 四态决策、人工审批要求与已知限制。

报告按 tenant 和 experiment_id 不可变存储；同 report_id 写入不同内容必须失败。趋势视图只能读取历史报告，不得覆盖原始结论。

## 11. 建议运营 SLA

以下 SLA 是团队运营起点，不是代码中的发布阈值：

| 事项 | 建议 SLA | 升级条件 |
|---|---|---|
| P0 安全退化初审 | 4 小时 | 任何新增安全失败或 veto |
| P1 主指标/关键切片退化归因 | 1 个工作日 | 通过率下降、关键能力切片显著退化 |
| Judge 错误或不可比性阻断 | 1 个工作日 | promotion 无法形成有效证据 |
| 回归候选人工审核 | 2 个工作日 | diagnostic 已生成 proposed 候选 |
| 发布审批材料复核 | 1 个工作日 | 状态为 eligible_for_human_approval |
| 周度趋势复盘 | 每周 | 连续两次 keep_baseline 或安全告警增加 |

每个问题应有 owner、状态、截止时间和关闭证据。SLA 超时只触发升级，不允许自动降低门禁阈值。

## 12. 验收标准

- 相同输入、配置与随机种子产生一致统计结果和稳定报告；
- 不可比输入被明确阻断，不生成伪精确的版本结论；
- diagnostic 永远不能获得发布资格，promotion 永远不能自动生成调优建议；
- 新增安全失败、一票否决或 P0 退化不会被平均分提升掩盖；
- 小样本切片被标记为观察项，不单独触发正式门禁；
- Markdown 包含目标、三层门禁、KPI/CI、切片、Bad Case、RCA、建议、决策和限制，且不泄露 query、answer 或 trace；
- 所有 `eligible_for_human_approval` 都明确要求人工审批。
