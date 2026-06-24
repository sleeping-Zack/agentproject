# 智扫通机器人智能客服 Agent

这是一个面向扫地/扫拖机器人的 RAG + 多工具 Agent 项目，覆盖知识库问答、天气/环境适配、用户设备使用记录查询、个性化报告生成、工具注册、安全检查、会话记忆、可观测 trace、FastAPI 服务化和 Streamlit 演示界面。

## 功能

- RAG 知识库：从 `data/` 中的 PDF/TXT 构建 Chroma 向量库。
- 多工具 Agent：支持知识库检索、天气、用户位置、用户 ID、当前月份、使用记录和报告上下文切换。
- MCP 风格工具注册：导出工具 manifest，并通过 allowlist 控制工具权限。
- 可控数据服务：用户、城市、月份和天气从配置读取，避免随机输出。
- 安全与可观测：提示词注入拦截、敏感字段脱敏、请求/工具/RAG trace。
- 服务化交付：FastAPI API、Streamlit UI、Dockerfile、CI 和 pytest 测试。

## 环境

推荐 Python 3.10.x，本项目使用 Python 3.10.11 验证。

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 中配置 `DASHSCOPE_API_KEY`。

## 启动

加载知识库：

```powershell
python -m rag.vector_store
```

启动 Streamlit 演示：

```powershell
streamlit run app.py
```

启动 FastAPI 服务：

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

常用接口：

- `GET /health`
- `GET /tools/manifest`
- `POST /chat`
- `GET /traces/{request_id}`

## 测试与评测

```powershell
python -m pytest tests -q
python scripts/evaluate_rag.py
```

`evals/rag_golden.jsonl` 是小型 RAG golden set，用于证明检索链路可以被重复评估。

## 部署

```powershell
docker build -t sweeper-agent .
docker run --env-file .env -p 8000:8000 sweeper-agent
```

## 目录

- `agent/`：Agent 封装、工具、中间件、会话记忆。
- `api/`：FastAPI 服务入口。
- `rag/`：向量库加载、RAG 总结和评测辅助。
- `services/`：可替换的数据服务适配器。
- `safety/`：输入安全和日志脱敏。
- `observability/`：轻量 trace。
- `tests/`：单元测试和 Prompt 回归测试。
- `evals/`：RAG golden set。

## 作品说明

这个项目不是单纯 Prompt Demo，而是把 Agent 应用开发中常见的工程关注点落成代码：工具权限、MCP 风格元数据、RAG 可评测、会话状态、安全边界、可观测性、API 服务和容器化交付。
