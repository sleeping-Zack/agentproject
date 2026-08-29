# CI 质量门禁策略

门禁的唯一策略源是 `config/ci_quality_gates.yml`。PR CI 不再在工作流里复制阈值；命令行阈值只保留给临时诊断覆盖，正式结果必须记录策略版本和 profile。

## 1. 两种门禁语义

`offline_fixture` 面向冻结数据和确定性 Mock。它保护代码回归：样本数、分桶覆盖和批准基线都必须存在，固定高风险断言不允许回退。

`online` 面向真实模型、真实检索和更大的候选集。一般质量允许有限失败，高风险安全、引用和必需工具仍保持零容忍。线上 profile 不能冒充 PR Fixture 门禁，两类报告会记录不同的 `evaluation_mode`。

## 2. 检索门禁

30 条冻结检索集要求 Recall@5 ≥ 0.90、Precision@5 ≥ 0.30、MRR ≥ 0.90、nDCG@5 ≥ 0.90、Hit Rate ≥ 0.90。

绝对阈值是灾难性退化的底线，不是允许回退的目标。冻结 Fixture 对批准基线实行所有指标零回退，因此 Hybrid 的实际下限是当前基线：Recall@5 0.9333、Precision@5 0.3933、MRR 0.9333、nDCG@5 0.9035、Hit Rate 0.9333。这样既解决了 Precision 0.15 过低的问题，也消除了 nDCG 只看 0.90 所造成的语义误解。

## 3. 生成门禁

当前 12 条固定 Fixture 全部标记为 critical，并显式覆盖四类：6 条质量、2 条安全、2 条引用/grounding、2 条拒答。离线回归要求：

- critical case pass rate、拒答准确率、事实覆盖率、引用有效率均为 100%；
- 危险指令率、禁止事实逃逸率、无依据声明率和 Judge 错误率均为 0；
- 任一类别样本被误删时，样本覆盖门禁直接失败。

真实模型 profile 至少需要 100 条样本。一般质量采用 Pass Rate ≥ 90%、事实覆盖率 ≥ 90%、Citation Validity ≥ 98%、Unsupported Claim Rate ≤ 5%、拒答准确率 ≥ 95%、Judge Error Rate ≤ 2%；critical case、critical citation、危险指令和禁止事实仍为零容忍。

## 4. Agent 门禁

所有比例只统计适用样例，不适用值为 `null` 并排除分母。当前 62 条 Fixture 的实际分母为：工具 40、参数 26、引用 18、关键词 35、artifact 35。没有预期工具的 22 条样例不再以满分稀释 Tool Recall。

总体要求 Pass Rate ≥ 90%、Tool Recall ≥ 90%、Parameter Accuracy ≥ 90%、Citation Validity ≥ 98%。同时执行风险分层：

- 高风险 33 条，覆盖 approval、budget、citation、report、security、tenant；其 Pass Rate 必须 100%；
- 高风险必需工具 20 条，Tool Recall 必须 100%；高风险适用参数和引用同样必须 100%；
- 普通必需工具 20 条，Tool Recall ≥ 90%；
- 11 个业务分桶各自有最低样本数，不能用增加普通样例来掩盖核心分桶缺失。

离线 Agent 报告中的耗时已改名为 `offline_harness_latency`，只表示 Mock backend 下控制面回归，P95 ≤ 500 ms；它不再被描述为真实端到端延迟。

## 5. 部署环境性能门禁

真实性能使用独立的 `.github/workflows/performance.yml`，只对已部署的流式 API 手动运行，不阻塞普通 PR。`config/api_performance_gates.yml` 明确记录请求数、并发数、超时和预热请求，并门禁：

- 成功请求数和成功 QPS；
- 首 Token P95；
- 完整响应 P95/P99；
- 超时率和总失败率；
- 服务指标窗口内模型阶段与工具阶段的平均耗时及 P95 桶上界。

性能工作流从 `performance` Environment 读取 `PERFORMANCE_TARGET_URL` 和 `PERFORMANCE_API_KEY`，保存完整 JSON artifact。当前 staging 配置是初始容量合同；变更阈值时必须附带同环境负载报告，不能用离线 Harness 数据代替。

## 6. CI 完整性

PR CI 启动 PostgreSQL 16 和 Redis 7.4，安装 production 依赖，并向测试进程提供集成测试连接。因此数据库、审批、artifact、人工评测、分析存储、事件总线和限流的分布式测试与其余测试在同一次全量测试中执行，不再因缺少服务而跳过。
