# Agent Quality Dataset v1

数据集版本：`agent-quality-v1`
Rubric 版本：`agent-rubric-v1`
状态：`candidate_pending_human_review`

该数据集是阶段二的端到端 Agent 评测规格集。每条记录定义输入、上下文、场景、风险、预期
行为、预期工具、关键事实、禁止事实和来源；不包含模型答案或人工评分。模型输出采集、双人
盲评和仲裁属于阶段三。

## 1. 为什么不直接合并旧 Golden

现有检索、生成和 Agent 文件合计约 194 条记录，但标准化后只有 136 个唯一问题，而且分布
明显偏向 RAG。部分 Agent 案例还是“生成第 1 份报告”“模拟后端异常场景 1”这样的确定性
回归占位符。

本数据集没有把这些重复或占位记录包装成新样本：

- 50 条真实知识/故障问题从 `evals/rag_golden.jsonl` 迁移，并补充端到端工具、风险、引用和
  来源要求。
- 工具、多步骤、安全、异常恢复和对抗边界案例按真实用户表达重新设计。
- 每条记录都有独立 `case_id` 和 `family_id`；语义家庭不能跨 split。
- 旧文件继续承担原有确定性回归职责，不被本阶段覆盖。

## 2. 数据产物

| 文件 | 用途 |
|---|---|
| `evals/agent_quality/schema_v1.json` | 单案例 JSON Schema |
| `evals/agent_quality/v1/dev.jsonl` | Prompt、策略和评测规则调试，只能使用该集合调优 |
| `evals/agent_quality/v1/test.jsonl` | 最终晋级判断，禁止用于调参 |
| `evals/agent_quality/v1/regression.jsonl` | 稳定回归与 CI 候选集合 |
| `evals/agent_quality/v1/manifest.json` | 版本、哈希、来源、split 策略和发布状态 |
| `evals/agent_quality/v1/coverage_report.json` | 场景、风险、难度、能力和来源覆盖 |
| `scripts/build_agent_quality_dataset.py` | 确定性构建入口 |
| `scripts/validate_agent_quality_dataset.py` | 完整性、重复、泄漏、覆盖和哈希校验 |

生成文件不应手工修改。变更案例蓝图或迁移规则后重新运行构建器，并同步更新版本记录。

## 3. 数据规模与分层

| 维度 | 分组 | 数量 |
|---|---|---:|
| split | dev / test / regression | 75 / 62 / 38 |
| 类别 | 知识与故障 | 50 |
| 类别 | 工具使用 | 30 |
| 类别 | 多步骤任务 | 25 |
| 类别 | 安全 | 25 |
| 类别 | 异常恢复 | 20 |
| 类别 | 对抗与边界 | 25 |
| 难度 | D1 / D2 / D3 | 10 / 117 / 48 |
| 风险 | L0 / L1 / L2 / L3 | 7 / 85 / 47 / 36 |

175 条案例对应 175 个语义家庭。七个 Rubric 主场景全部覆盖：知识问答、故障诊断、工具
执行、多步骤任务、澄清与拒答、安全边界、异常与降级。

test 比例高于默认 20%，原因是历史冻结检索 test 及其人工审计同意图改写被强制归入新
test。这是有意的防泄漏约束，不应为了调整比例将这些问题移动到 dev。

## 4. 单条记录语义

核心字段：

- `case_id`：版本内唯一案例标识。
- `family_id`：同意图或参数变体的语义家庭；同一家庭必须处于同一 split。
- `category`：阶段二的覆盖类别。
- `scene`：阶段一 Rubric 的唯一主场景。
- `query` / `turns`：当前问题和可选多轮上下文。
- `context`：身份、租户、日期、权限、预算或故障注入条件。
- `labels`：能力、难度和风险标签。
- `expected.behavior`：可由评测员执行的期望行为，不要求唯一措辞。
- `expected.tools`：预期工具、参数和参数匹配方式。
- `expected.facts` / `forbidden_facts`：必须覆盖和禁止出现的内容。
- `references`：语料、工具 Schema、策略或规范的本地可验证来源。
- `provenance`：构造方式、原始案例和审核状态。

`argument_match=contains` 表示只校验列出的必要参数字段，允许 Agent 补充合法的
`information_gap`；`exact` 表示参数对象应完全一致。`$tool:tool_name` 表示参数来自前一步
真实工具结果，不能写死或猜测。

## 5. Split 与防泄漏规则

默认使用固定种子 `20260810` 按 `family_id` 分组分配 60% dev、20% test、20% regression。
以下规则优先于默认比例：

1. `evals/retrieval_golden.jsonl` 中的冻结查询只能进入 test。
2. `evals/retrieval_test_query_aliases_v1.jsonl` 中的人工审计同意图改写只能进入 test。
3. 同一 `family_id` 不得跨 split。
4. 标准化后完全相同的 query 直接报错。
5. 字符三元组 Jaccard 不低于 0.90 的近重复若跨 split，直接报错；同 split 产生警告。
6. manifest 对每个数据文件、来源文件和全数据集保存 SHA-256；源文件变化后旧 manifest
   自动失效。

test 只能用于最终晋级判断。观察到 test Bad Case 后，应在下一版本建立新的 dev/regression
案例进行修复，不得根据原 test 文本调 Prompt、工具策略或评分阈值。

## 6. 构建与验证

```powershell
.\.venv\Scripts\python.exe -m scripts.build_agent_quality_dataset
.\.venv\Scripts\python.exe -m scripts.validate_agent_quality_dataset
.\.venv\Scripts\python.exe -m pytest tests\test_agent_quality_dataset.py -q
```

验证器检查：

- 必需字段、枚举、工具名、能力标签和本地引用。
- query、case ID 和 family 的重复/泄漏。
- 冻结检索 test 及 alias 隔离。
- 类别目标、七场景覆盖、D3 和 L3 最低数量。
- citation/artifact 要求与能力标签一致。
- 数据、覆盖报告、构建器、Rubric 和来源文件哈希。

## 7. 审核边界

所有案例当前均为 `pending_second_reviewer`，manifest 明确设置
`production_golden_allowed=false`。这意味着：

- 可以用于阶段三的试标、标注员校准和流程开发。
- 在独立人员复核期望行为和完成仲裁前，不能宣称为“人工高质量 Golden”。
- 不能以当前构造者自审代替双人盲评。
- 发现歧义时应修订 dev 案例及规范；冻结 test 的变更必须升级数据集版本。

阶段三晋级至少需要：第二评审完成、争议案例仲裁、Rubric 一致性达标、每条最终标签可追溯，
并将 manifest 状态升级为经人工批准的版本。
