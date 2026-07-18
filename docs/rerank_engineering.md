# Rerank 工程改造与晋级说明

## 1. 当前结论

旧链路让 BGE 直接覆盖 RRF 排名，在 30 条冻结 test 查询上使 Recall@5 从
`0.9333` 降到 `0.7389`，MRR 从 `0.9333` 降到 `0.8389`，因此不能直接用
Cross-Encoder 替换 Hybrid 排名。

本轮使用 25 条已审核 dev 查询选择保守权重：Hybrid `0.9`、BGE `0.1`。真实模型结果为：

- dev：Recall@5 `0.2739 -> 0.2755`，nDCG@5 `0.6037 -> 0.6066`，MRR 保持 `1.0`，
  没有逐查询 Recall 或排序退化。
- 冻结 test：Recall@5 `0.9333`、MRR `0.9333`、nDCG@5 `0.9035`，与 Hybrid
  完全持平，没有逐查询退化，也没有新增提升。
- CPU 远程服务：dev 平均 `10.135s`、P95 `11.632s`；test 平均 `10.570s`、
  P95 `12.177s`，未通过 `P95 <= 1s` 门禁。

因此新策略解决了旧 rerank 的质量退化，并在 dev 上有小幅提升，但尚未获得可上线的延迟，
冻结 test 也没有证明额外收益。生产默认继续 `enable_reranker: false`；实验开启时使用
`shadow`，不得直接切到 `on`。

## 2. 已实施

- BGE 输入由纯正文改为标题、型号、章节、版本、页码和正文组成的结构化 passage。
- 文档 metadata 增加稳定 `document_title`，Markdown chunk 继承最近章节标题。
- 保留 Dense、BM25、RRF、BGE 和最终 `ranking_score`，用于逐候选诊断。
- 扩大候选池时锚定原 RRF Top20，避免新尾部候选破坏已验证的头部顺序。
- 使用排名融合而非直接混加不同尺度的原始分数。
- 型号、错误码和数值约束查询绕过 reranker，优先保留 BM25/RRF。
- 支持 `shadow`、`weighted_rrf` 和旧版 `replace` 三种策略。
- 模型未返回完整分数、推理失败、超时或断路器打开时保留 Hybrid 排名。
- 评测报告包含候选池 Recall、逐查询退化 ID、平均/P95 延迟和发布门禁。
- 锁定 30 条 test 及 8 条同意图改写；调参脚本在代码层拒绝非 dev 数据。
- 25 条 dev 共 858 个候选已完成审核，并生成 dev golden 与 hard negatives。
- 三组 chunk 参数使用独立 collection、Chroma、MD5 和 BM25 存储。
- 已实现远程客户端和真实 FastAPI BGE 服务，包含预加载、健康检查、输入边界和完整分数契约。

## 3. 默认配置

```yaml
retrieval:
  enable_reranker: false
  reranker_backend: local
  reranker_url: http://127.0.0.1:8090/rerank
  reranker_timeout_seconds: 2.0
  reranker_failure_threshold: 3
  reranker_recovery_seconds: 30.0
  fusion_anchor_k: 20
  reranker_model: BAAI/bge-reranker-v2-m3
  rerank_version: bge-policy-v3
  rerank_strategy: shadow
  rerank_hybrid_weight: 0.9
  rerank_model_weight: 0.1
  rerank_fusion_k: 10
  rerank_bypass_exact_queries: true
  rerank_max_document_chars: 1200
```

等价环境变量位于 `.env.example`。修改权重、模型、输入、路由或排序语义时必须递增
`rerank_version`，避免语义缓存复用旧结果。`2s` 是线上失败保护，不是本机 CPU 离线评测
超时；离线评测显式使用 `60s`。

## 4. 数据与调参

构建已审核 dev 与 hard negatives：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_rerank_dev_data
```

启动 shadow 评测并只在 dev 上扫描权重：

```powershell
$env:AGENT_RERANK_ENABLED="true"
$env:AGENT_RERANK_BACKEND="remote"
$env:AGENT_RERANK_URL="http://127.0.0.1:8090/rerank"
$env:AGENT_RERANK_TIMEOUT_SECONDS="60"
$env:AGENT_RERANK_STRATEGY="shadow"
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval `
  --golden evals\retrieval_dev_golden.jsonl --split dev `
  --enable-reranker --candidate-k 20 `
  --report reports\retrieval-rerank-dev-shadow.json

.\.venv\Scripts\python.exe -m scripts.tune_rerank_fusion
```

权重扫描结果中，`0.10` 是满足“无逐 case Recall 退化”约束下 nDCG 最高的 BGE 权重。
从 `0.15` 开始，`rag-aac997b3d482` 出现 Recall 退化，因此不采用更激进权重。

## 5. 冻结测试门禁

参数从 dev 选定后，冻结 test 只执行一次，不根据结果反向调参：

```powershell
$env:AGENT_RERANK_STRATEGY="weighted_rrf"
$env:AGENT_RERANK_HYBRID_WEIGHT="0.9"
$env:AGENT_RERANK_MODEL_WEIGHT="0.1"
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval `
  --golden evals\retrieval_golden.jsonl --split test `
  --enable-reranker --candidate-k 20 `
  --baseline evals\baselines\retrieval_baseline_v1.json `
  --report reports\retrieval-rerank-test-weighted-v3.json `
  --gate --gate-strategy hybrid_rerank `
  --min-recall 0.9333 --min-mrr 0.9333 --min-ndcg 0.9035 `
  --min-hit-rate 0.9333 --min-candidate-recall 0.9333 `
  --max-recall-regressed-cases 2 --max-p95-latency-ms 1000
```

该门禁只有 `p95_latency_above_threshold` 失败。质量指标全部达到 Hybrid 基线，但未在 test
上超过 Hybrid。

## 6. Chunk 隔离实验

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_chunk_experiments
```

`200/20`、`350/50`、`500/80` 三组索引已构建并通过 manifest 校验。Chroma 与 BM25
文档数分别为 `326/326`、`185/185`、`135/135`，每组 BM25 指纹均与对应 Chroma 一致。

现有 858 个标签只对应 `200/20` 的 doc_id。chunk 变化会改变文本边界和 doc_id，因此若要
比较另外两组的质量，必须分别生成并审核候选，不能复用当前标签。

## 7. 真实 Rerank 服务

安装 rerank 可选依赖并启动单进程服务：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[rerank]"
$env:AGENT_RERANK_SERVICE_PRELOAD="true"
.\.venv\Scripts\python.exe -m uvicorn api.reranker_server:app `
  --host 127.0.0.1 --port 8090
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
Invoke-RestMethod http://127.0.0.1:8090/ready
```

客户端发送：

```json
{"model":"BAAI/bge-reranker-v2-m3","query":"...","documents":["结构化 passage"],"top_n":20}
```

服务返回与输入顺序对齐的 `scores`。它拒绝运行时切换模型、空文档、超长文档和不完整
`top_n`，并在启动时预加载模型。客户端继续负责超时、响应校验、指标、断路器和 Hybrid
回退。每个 Uvicorn worker 都会加载一份约 2.29 GB 模型，因此当前配置应保持单 worker；
GPU 多副本、动态批处理和容量压测必须在有 CUDA 的部署环境完成。

## 8. 发布流程

发布顺序固定为 `off -> shadow -> canary -> on`：

- `off`：当前默认，不调用 reranker。
- `shadow`：记录 BGE 排名，不改变用户结果。
- `canary`：只在 GPU 服务达到质量、P95、失败率和容量门禁后开放小流量。
- `on`：扩大流量；任一门禁回退立即恢复 Hybrid。

当前停留在 `off/shadow`。剩余阻塞是 GPU 环境中的延迟与容量门禁，而不是标签、索引、
模型调用契约或离线质量回归。
