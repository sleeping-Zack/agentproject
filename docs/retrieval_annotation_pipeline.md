# Retrieval 标注与 Rerank Dev 流水线

候选发现、人工判断、dev 构建、调参和最终 test 验证必须分开。检索分数、答案关键词和
来源文件名都不能自动充当相关性标签。

## 1. 锁定测试集

`evals/retrieval_golden.jsonl` 中现有 30 条数据全部是冻结 test。它只用于最终回归和晋级
判断，不用于模型选择、权重扫描、路由规则设计或 chunk 参数选择。

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_retrieval_manifest
```

该命令校验语料文件 SHA-256、chunk、Embedding 和检索版本，防止用错索引评测。

## 2. 生成独立 dev 候选

```powershell
$env:AGENT_RERANK_ENABLED="false"
.\.venv\Scripts\python.exe -m scripts.generate_retrieval_golden `
  --output evals\annotations\retrieval_dev_candidates_v1.jsonl `
  --top-k 20 --timeout 60
```

脚本默认读取 `evals/rag_golden.jsonl`，排除冻结 test 的规范化 query，并额外读取
`evals/retrieval_test_query_aliases_v1.jsonl` 排除 8 条人工审计的同意图改写。没有 ID 的
输入会按 query 生成稳定哈希 ID。输出记录固定为 `split: "dev"`、
`review_status: "pending"`，并保存 Dense、BM25、Hybrid 的候选正文、metadata、排名、
分数和版本。

当前 25 条 dev、858 个候选均已完成审核并导入版本化标签。候选原始文件仍保留
`pending`，最终判断以 `retrieval_dev_labels_v1.jsonl` 为准，避免把发现阶段与标签阶段混写。

## 3. 人工四级标注

把审核结果写入 `evals/annotations/retrieval_dev_labels_v1.jsonl`。每行格式如下：

```json
{"case_id":"rag-...","query":"...","split":"dev","labels":[{"doc_id":"...","grade":3,"rationale":"直接回答问题"},{"doc_id":"...","grade":0,"rationale":"同词但产品或结论不匹配"}],"review_status":"reviewed","reviewed_by":"reviewer-name"}
```

等级含义：

- `0`：不相关，可作为经过人工确认的 hard negative
- `1`：主题相关但不能回答，不自动当作训练正例或负例
- `2`：部分回答，可作为 reranker 训练正例
- `3`：直接回答，可作为 reranker 训练正例

每个标签都必须有 rationale。不要把“未标注候选”自动视为负例。

## 4. 构建 dev 与 hard negative

审核完成后运行：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_rerank_dev_data
```

脚本再次检查所有记录是 dev、query 不与冻结 test 重叠、标签 doc_id 确实来自对应候选池，
然后生成：

- `evals/retrieval_dev_golden.jsonl`：只含 dev，用于模型选择和调参
- `evals/training/rerank_hard_negatives_v1.jsonl`：`grade>=2` 正例与显式 `grade=0` 负例

当前产物包含 25 条 golden case；训练数据包含 131 个 `grade>=2` 正例和 111 个经过审核的
`grade=0` hard negative。golden 的检索相关集合使用 `grade>0`，训练正例使用更严格的
`grade>=2`，两者用途不同。

旧的 `split_retrieval_labels.py` 只保留给早期通用标注流程；本轮 rerank 实验不要用它覆盖
冻结的 `evals/retrieval_golden.jsonl`。

## 5. Shadow 评测与 dev-only 调参

```powershell
$env:AGENT_RERANK_ENABLED="true"
$env:AGENT_RERANK_STRATEGY="shadow"
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval `
  --golden evals\retrieval_dev_golden.jsonl `
  --enable-reranker --candidate-k 20 `
  --report reports\retrieval-rerank-dev-shadow.json

.\.venv\Scripts\python.exe -m scripts.tune_rerank_fusion
```

调参脚本只接受 `split=dev`，并要求权重网格包含纯 Hybrid 基线 `0.0`。推荐权重必须没有
任何逐 case Recall 退化，再按 nDCG、MRR 和更低 BGE 权重依次选择。本轮推荐结果为
Hybrid `0.9`、BGE `0.1`；dev 有小幅提升，冻结 test 与 Hybrid 持平。

## 6. 最终 test 晋级

参数冻结后只在 30 条 test 上运行一次门禁。失败时回到 dev 分析，不根据 test case 再调权重。
完整阈值和发布顺序见 `docs/rerank_engineering.md`。
