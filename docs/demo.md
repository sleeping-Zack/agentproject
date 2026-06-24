# 真实演示说明

## 启动

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

默认开发 API Key 是 `dev-api-key`，生产环境通过 `AGENT_API_KEY` 覆盖。

## 示例 1：健康检查

```bash
curl http://127.0.0.1:8000/health
```

响应：

```json
{"status":"ok"}
```

## 示例 2：工具 Manifest

```bash
curl http://127.0.0.1:8000/tools/manifest
```

响应会包含 MCP 风格工具元数据，例如 `rag_summarize`、`get_weather`、`fetch_external_data`。

## 示例 3：普通客服问答

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d "{\"message\":\"主刷缠绕毛发怎么办？\",\"session_id\":\"demo-qa\"}"
```

预期行为：系统会进入 Agent 链路，必要时调用 RAG 工具，并返回带引用来源的处理建议。

## 示例 4：个人使用报告

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d "{\"message\":\"帮我生成本月使用报告\",\"session_id\":\"demo-report\"}"
```

预期行为：API 会走显式 ReportWorkflow，固定执行意图识别、用户上下文、使用记录查询、RAG 补充建议和报告生成。

## 示例 5：MCP HTTP 调用

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"深圳\"}}}"
```

响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {"type": "text", "text": "城市深圳天气为多云，气温30摄氏度，空气湿度72%，南风2级，AQI28，最近6小时降雨概率低"}
    ]
  }
}
```

## Trace 示例

```bash
curl http://127.0.0.1:8000/traces/<request_id>
curl http://127.0.0.1:8000/traces/<request_id>/otel
```

Trace 中包含 `request_id`、`session_id`、工具/模型/Agent span、耗时、参数摘要和错误信息。OTel endpoint 会导出 OpenTelemetry 风格 span，方便后续接入真实链路追踪平台。
