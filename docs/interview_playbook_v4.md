# Agent 项目面试讲稿 · 第四轮：生产级工程化升级

> 这份讲稿独立于 `interview_playbook.md`，专门讲第四轮升级。背诵时把每节"亮点"与"被追问怎么答"记下来，足以撑起 30~40 分钟的技术深聊。

## 0. 总览（30 秒电梯讲）

> "我把这个 Agent 项目按多租户 Agent 平台的工程标准做了系统升级：真实 Hybrid RAG、可复现评测、统一预算、外部化策略、事实验证、幂等持久化、完整语义缓存和可重放 SSE。每个能力都有确定性测试和 CI 门禁，而不是只靠演示。"

一句话亮点串场：
- **"健康度自动选模型 + 失败降级"**（#1 #2 联动）
- **"一条日志 = 一条 JSON，自动带 request_id/tenant/prompt_version"**（#3 #5 联动）
- **"前端能看到 Agent 在调哪个工具，因为我做了 SSE 事件总线"**（#6）
- **"工具命中率 + 关键词命中率，CI 自动跑"**（#4）
- **"语义缓存 + 幂等缓存 + prefix hint 三层，命中率上 metrics"**（#9）

---

## 1. 多模型路由 & 国产模型适配

### 痛点
原来 `model/factory.py` 写死 `ChatTongyi`，换豆包要改一片代码。生产环境一定是多家模型并存：豆包做主、通义做备、本地 vLLM 做长上下文兜底。

### 实现要点
- `model/providers.py` 抽象出 `ProviderConfig`，每个 provider 类只实现 `as_langchain_model()`。
- 新增 **DoubaoProvider / OpenAICompatibleProvider / VLLMProvider**。火山方舟（豆包）原生兼容 OpenAI Chat Completions，所以这三个 provider 共享同一个 `ChatOpenAI` 客户端，只是 `base_url` 与 `api_key_env` 不同。
- `model/router.py` 引入 `ModelRouter`：每个 provider 注册成 `ProviderEntry`（带 scene / tenants / weight / breaker），`invoke(fn)` 按健康度选主、失败自动 fallback。

### 关键代码

```python
# model/providers.py
class DoubaoProvider(OpenAICompatibleProvider):
    name = "doubao"
    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        if not config.api_key_env:
            config.api_key_env = "ARK_API_KEY"
        super().__init__(config)

# model/router.py
def invoke(self, fn, scene="default", tenant_id=None):
    candidates = self._candidates(scene, tenant_id)
    for entry in candidates:
        if not entry.breaker.allow():
            continue
        try:
            with bind_request_context(model=entry.name):
                model = build_model_provider(entry.config).as_langchain_model()
                result = fn(model)
            entry.breaker.record_success()
            return result
        except Exception as exc:
            entry.breaker.record_failure()
            continue
    raise NoAvailableModelError(...)
```

### 面试亮点
1. **OpenAI-Compatible 是一招吃三家**：豆包、vLLM、Together 都走同一份 `ChatOpenAI`，新增 provider 只多写 10 行默认配置。
2. **路由策略可扩展**：`scene`（默认/长上下文/低成本）、`tenants`（白名单）、`weight`（权重）三维选择。
3. **健康度由熔断器提供**：和 #2 解耦，router 只 `entry.breaker.allow()` 一行。
4. **选中的 model 名字进 ContextVar**：日志里每条都能看到这次实际打到哪家。

### 可能被追问
- **Q：为什么不直接用 LiteLLM 这种统一网关？**
  A：LiteLLM 是个不错的选择，但我做这个项目想把控制点都留在自己手里——熔断、租户隔离、prompt 版本注入这几件事都需要 router 直接配合，外部网关会增加调试链路长度。生产环境我会评估，体量上来后大概率会迁。
- **Q：fallback 顺序怎么定？**
  A：先按 scene 匹配（精确 > default），再按 weight 降序。健康的优先于熔断打开的；都熔断时也会按顺序试一次半开探测，避免永久不恢复。
- **Q：如果备用模型也挂了？**
  A：抛 `NoAvailableModelError`，上层在 `execute_stream` 里被 catch 后 metrics 打 `status="error"`，并向用户返回固定兜底文案。这层兜底放在 `react_agent.py` 而不是 router，是为了让 router 只关心"能不能跑通"，不掺业务文案。

---

## 2. 熔断器（半开/全开/关闭三态）

### 痛点
之前只有重试（`RetryPolicy`）和限流（`RateLimiter`），中间这一层缺失：一个工具持续失败时整条 Agent 链路会被反复拖累。

### 实现要点
- `services/circuit_breaker.py` 实现标准三态机：`CLOSED → OPEN → HALF_OPEN → CLOSED`。
- `failure_threshold` 连续失败次数阈值；`recovery_timeout` OPEN 后多久允许半开探测；`half_open_max_calls` 半开期允许试探次数。
- 配套 `CircuitBreakerRegistry` 按 name 单例管理，方便在 metrics 里聚合。
- **接入两处**：
  - `model/router.py` 用每个 provider 自带的 breaker 做"健康度过滤"。
  - `agent/tools/middleware.py` 用 `breaker_registry.get(f"tool:{tool_name}")` 保护工具调用，熔断打开时直接返回 `ToolMessage` 兜底文案。

### 关键代码

```python
# services/circuit_breaker.py
def allow(self) -> bool:
    with self._lock:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
            else:
                return False
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
```

### 面试亮点
1. **为什么自己写不用 pybreaker**：状态变化需要直接 hook `metrics_registry`，自己写一百多行更可控。每次状态切换都 `inc_counter("agent_circuit_state_transition_total")`，Grafana 一眼能看到。
2. **半开探测限并发**：`half_open_max_calls=1` 防止一拥而上把刚恢复的服务再拍死。
3. **工具熔断的兜底语义**：返回的不是异常，是 `ToolMessage(content="工具…当前不可用…")`，让 Agent 继续推理（"换种方式回答用户"），而不是整条链路崩。

### 可能被追问
- **Q：熔断和限流的关系？**
  A：限流防过载（"你别打我太多次"），熔断防级联（"我现在打不动你了，别浪费"）。两者是正交的，串行执行：先限流过滤再熔断保护。
- **Q：怎么调阈值？**
  A：起步给 `failure_threshold=5, recovery_timeout=30s`，再看 metrics 里 `agent_circuit_state_transition_total` 的频率。如果切换过于频繁说明阈值太敏感，如果用户已经感知到错误才切换说明阈值太钝。
- **Q：本地内存熔断在多实例部署下会不一致？**
  A：是。生产环境每个实例独立熔断是可以接受的折中（最多比理想晚 N 实例 × threshold 才整体降级）。要全局一致就接 Redis 共享计数，但增加了 RTT。

---

## 3. 结构化日志 + TraceID 贯穿

### 痛点
原来 `logger_handler` 输出纯文本，多服务串联调试时只能凭时间戳猜对应关系。

### 实现要点
- 新增 `observability/context.py`：`RequestContext` 数据类 + `ContextVar` + `bind_request_context()` 上下文管理器 + `update_request_context()` 就地更新。
- `utils/logger_handler.py` 新增 `JsonFormatter`：每条 log 输出一行 JSON，自动注入当前 ContextVar 中的字段。
- 环境变量 `AGENT_LOG_FORMAT=text|json` 切换格式，默认 json。

### 关键代码

```python
# observability/context.py
@contextmanager
def bind_request_context(**fields):
    current = _current.get()
    merged = RequestContext(
        request_id=fields.get("request_id", current.request_id),
        session_id=fields.get("session_id", current.session_id),
        tenant_id=fields.get("tenant_id", current.tenant_id),
        model=fields.get("model", current.model),
        prompt_version=fields.get("prompt_version", current.prompt_version),
        ...
    )
    token = _current.set(merged)
    try:
        yield merged
    finally:
        _current.reset(token)

# utils/logger_handler.py
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat(...) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "file": f"{record.filename}:{record.lineno}",
            "msg": redact_sensitive(record.getMessage()),
        }
        payload.update(request_context().as_dict())  # ★ 关键一行
        return json.dumps(payload, ensure_ascii=False)
```

### 面试亮点
1. **ContextVar 而不是 threading.local**：`ContextVar` 跨 asyncio 任务自然传播，未来异步化也不用改。
2. **bind / update 双 API**：
   - `bind_request_context()` 是栈式（with 块退出会 reset），适合请求入口。
   - `update_request_context()` 是就地写入，适合"prompt 加载后才知道 prompt_version"这种延后字段。
3. **可观测三件套联动**：`/metrics` 看趋势 → trace 看单次 → 结构化日志看上下文，三者通过 `request_id` 串成完整故事。

### 可能被追问
- **Q：JSON 日志会不会让本地调试很难看？**
  A：所以我留了 `AGENT_LOG_FORMAT=text`，本地 dev 切回原格式。生产固定 json 灌 ES / Loki，CLI 想看人工友好就 `jq` 一下。
- **Q：敏感字段怎么防泄露？**
  A：`redact_sensitive` 在 message 层级先脱敏（API key、token、邮箱模式正则），再交给 JsonFormatter 序列化。如果有自定义字段直接传到 record，我们做了 try/except 来确保不可 JSON 序列化的值不会让整条日志失败。

---

## 4. 端到端 Agent 评测

### 痛点
原来只有 `evals/rag_golden.jsonl` 评 RAG 单点检索，**没有评 Agent 整链路**——工具有没有被正确调用、多轮上下文记没记住、prompt injection 有没有挡住，统统靠人工试。

### 实现要点
- `evals/agent_offline_golden.jsonl` 固定 62 条，覆盖 RAG、工具路由与参数、报告审批、安全、异常降级、引用正反例、预算、多轮、租户和缓存。
- PR 使用 scripted backend 跑完整 `AgentRunner`，不是只校验 JSON：策略、预算、Verifier、artifact 和状态机都真实执行。
- 指标包含 pass rate、tool recall、parameter accuracy、keyword recall、citation validity、artifact save rate、P95 和成本。
- 同时执行固定阈值和 `agent_baseline_v1.json` 相对退化检查；真实模型评测放在独立定期/手动 workflow，不阻断 PR。

### 关键代码

```python
# scripts/evaluate_agent.py
runner = _OfflineRunnerFactory(cases).build(case)
result = runner.run(AgentTask(query=query, tenant_id="eval", ...))
parameter_accuracy = _tool_parameter_accuracy(expected_tools, result.state.tool_calls)
gate = EvalGate(thresholds).evaluate(report)
baseline = _compare_baseline(report["aggregate"], report["latency"], baseline_path)
```

### 面试亮点
1. **评测拉链贯穿三层**：golden set 定义预期 → trace 提供观察口 → 脚本算指标。三者解耦，加 case 只动 jsonl。
2. **CI 跑完整确定性链路**：不消耗线上 token，但不是 dry-run；62 条全部通过后还要满足 baseline delta。
3. **报告可机读**：`--report path.json` 输出结构化报告，方便 `scripts/prompt_diff.py` 抓取做 A/B。

### 可能被追问
- **Q：关键词命中率会不会太弱，模型换种说法就过不了？**
  A：是的，所以这只是第一层粗筛。下一步接 `rag/judge.py`（LLM-as-judge）做语义层评分，但要烧 token，所以放线下/批量跑。这套 keyword 评测在 CI 里能跑、几秒出结果，是不同 trade-off。
- **Q：工具命中率怎么算"调对了"？**
  A：工具名和参数分开计分；`parameter_accuracy` 会逐字段比对 golden 中的 `expected_tools[i].args`，避免“工具选对但参数错”仍被算通过。
- **Q：多轮 case 怎么写？**
  A：jsonl 里 `turns` 字段是历史轮次，最后一条 user 触发评测；评测前用 `agent.memory.add_message` 灌入历史，复用 multi-tenant 通道（`tenant_id="eval"`）保证不污染线上 session。

---

## 5. Prompt 版本管理

### 痛点
prompt 改一次就是一次"没人知道动了什么"的迭代。线上效果回退也找不到对照。

### 实现要点
- 三个 prompt 文件加 YAML frontmatter：
  ```yaml
  ---
  version: v2
  changelog:
    - "v2: 强化报告生成强约束……"
    - "v1: 初版"
  ---
  你是扫地机器人……（正文）
  ```
- `utils/prompt_loader.py` 解析 frontmatter，返回 `PromptDocument(content, version, changelog)`。
- 加载时通过 `update_request_context(prompt_version=...)` 写入 ContextVar，**结构化日志和 trace 自动捕获**。
- `scripts/prompt_diff.py`：传入 baseline 文件 → 临时替换 prompt → 跑评测 → 还原 → 输出 A/B 报告。

### 关键代码

```python
# utils/prompt_loader.py
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

def _parse(name, raw):
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return PromptDocument(name=name, content=raw)
    meta = yaml.safe_load(match.group(1)) or {}
    return PromptDocument(
        name=name,
        content=raw[match.end():],
        version=str(meta.get("version", "unversioned")),
        changelog=[str(x) for x in (meta.get("changelog") or [])],
    )

def _activate(doc):
    update_request_context(prompt_version=f"{doc.name}:{doc.version}")
    return doc.render()
```

### 面试亮点
1. **零侵入**：没有 frontmatter 的旧 prompt 视为 `unversioned`，向后兼容。
2. **版本进 trace**：线上 bug 时可以直接看 trace 里这次请求用了 `main:v2`，对应到 git history 找原 prompt 文件。
3. **prompt_diff 把"改 prompt 是不是更好"变成可量化命题**：备份 → 替换 → 评测 → 还原。

### 可能被追问
- **Q：版本号怎么管？手动递增？**
  A：现在是手动 v1/v2/v3。下一步把 frontmatter 中的 `version` 默认取 `git log -n 1 --format=%h prompts/main_prompt.txt`，自动绑定 commit hash。
- **Q：A/B 怎么真正在线跑？**
  A：在 `ProviderEntry` 那套机制上扩展即可——按 `tenant_id` hash 取模分流到 prompt v1 / v2，metrics 区分标签收口比较即可。架构留了口子。

---

## 6. SSE 流式增强（TTFT / 工具事件 / 心跳）

### 痛点
- 原 `/chat/stream` 只发 `data:` 单一事件，前端只能看到答案，不知道 Agent 在"调哪个工具"。
- 没有 TTFT 指标，无法回答"模型多久开始吐第一个字"。
- nginx 默认 60s 无数据就断连，长任务会被掐。

### 实现要点
- `AgentEvent` 固定携带 request_id、event_type、严格递增 sequence、timestamp 和脱敏 payload。
- EventBus 为每个 request 提供有界 live queue、短期 replay buffer、背压取消和 stream identity；同一 request 只能绑定同一 tenant/session/query。
- LangGraph `stream_mode=["messages", "updates"]` 的 message chunk 立即发布 `token_delta`，不等待 Agent 完成。
- Runner 与工具中间件发布 run/model/tool/approval/verifier/artifact/terminal 事件；长任务定时发送 heartbeat。
- SSE 使用标准 `id / event / data`，客户端以同一 request_id 和 `Last-Event-ID` 重放遗漏事件；断开时设置取消标志。
- 仅首个 `token_delta` 记录 TTFT，和总响应时间分开统计。

### 关键代码

```python
# AgentRunner
async for event in runner.run_stream(task, last_event_id=cursor):
    yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: ...\n\n"

# ReactAgent message stream
delta = self._message_text(message)
event_bus.publish(request_id, "token_delta", {"delta": delta, "provisional": True})

# Tool middleware only emits for streaming requests; args are redacted first
_publish_tool_event(request_id, emit_events, "tool_started", payload)
_publish_tool_event(request_id, emit_events, "tool_completed", payload)
```

### 面试亮点
1. **事件总线有明确可靠性边界**：进程内、有界、可重放；不会无限吃内存，也不会把慢客户端悄悄丢事件。
2. **同步执行与异步协议解耦**：Runner producer 放入工作线程，async generator 实时消费统一事件，心跳和取消不阻塞事件循环。
3. **TTFT 是 Agent 服务的关键 SLO**：用户体感上"多久看到第一个字"比"总耗时"更重要，我把它做成直方图指标。

### 可能被追问
- **Q：close() 时机？**
  A：Runner 发布 `run_completed/run_failed` 后，producer 的 `finally` 关闭 channel；replay buffer 在 retention 窗口内继续保留，重连不会重跑任务。
- **Q：多个客户端同 request_id 会怎样？**
  A：当前 live queue 是单消费者，目标是断线恢复而不是广播；identity 会阻止跨租户/跨 query 附着。多实例或多端订阅时应换 Redis Streams/NATS，并按 consumer/fanout 语义设计。

---

## 7. 多租户隔离

### 痛点
之前 `session_id` 是裸字符串，租户 A 和租户 B 用同一个 `session_id="default"` 就互相串了。限流维度也是 IP，单租户挤占整体配额。

### 实现要点
- API 入口：`X-Tenant-ID` header + `ChatRequest.tenant_id` body，两者都没传时 fallback `"default"`。
- `agent/memory.py`：所有读写方法加 `tenant_id` 参数，内部 key 为 `f"{tenant_id}|{session_id}"`。
- `services/persistence.py`：SQLite 两表 `session_messages` / `traces` 增加 `tenant_id` 列与索引，`ALTER TABLE` 自动迁移。
- `SessionStore` 协议用 `"tenant|session"` 串编码 session_id，向后兼容（没有 `|` 视为 `default`）。
- 限流：`_rate_limit(request, tenant_id)` 优先按 `tenant:xxx` 桶，匿名才退回 `ip:xxx`。
- ContextVar 注入 `tenant_id`，日志/trace 自动带上。

### 关键代码

```python
# api/server.py
def _rate_limit(request, tenant_id):
    if tenant_id and tenant_id != "default":
        key = f"tenant:{tenant_id}"
    else:
        client_host = request.client.host if request.client else "unknown"
        key = f"ip:{client_host}"
    if not rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

# services/persistence.py - 自动迁移老库
@staticmethod
def _ensure_column(conn, table, column, ddl):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
```

### 面试亮点
1. **三层全打通**：API → memory → DB → 限流 → 日志/trace 都按 tenant 分。
2. **演进式 schema migration**：`_ensure_column` 让老库能直接升级，不用写一次性脚本。
3. **限流维度**：tenant 优先，IP 兜底，匿名也不会无限制。

### 可能被追问
- **Q：SQLite 在多租户下扛得住吗？**
  A：演示足够，生产换 Postgres。Schema 我设计时就考虑了 `(tenant_id, session_id)` 复合索引，迁到 PG 直接照搬。
- **Q：跨租户的 RAG 怎么办？**
  A：当前共用一个向量库。要租户级隔离的两种做法：Chroma collection 按 tenant 拆，或 metadata filter（`where={"tenant_id": "..."}`）。前者隔离更强、容量管理简单；后者节省索引开销，但 retriever 不能漏过滤。

---

## 8. 异步化与并发

### 痛点
FastAPI 端点都是 `def chat(...)` 同步函数，每个请求占满一个 worker 线程，模型调用一阻塞，整体并发上不去。

### 实现要点
- 所有端点改 `async def`。
- Agent 调用是阻塞的（LangChain 同步），所以用 `asyncio.to_thread(_run)` 丢线程池，**不阻塞事件循环**——这样事件循环可以同时处理别的请求的 IO。
- `PlanExecutor.execute_async()` 新增 asyncio 路径：独立子任务 `asyncio.gather(*to_thread(...))`，比 ThreadPoolExecutor 省内存且与 FastAPI 事件循环共享。

### 关键代码

```python
# api/server.py
@app.post("/chat", response_model=ChatResponse)
async def chat(request, raw_request, x_api_key, x_tenant_id):
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(request.tenant_id, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    ...
    def _run():
        # 同步 Agent 调用
        chunks = list(agent.execute_stream(...))
        return get_final_response(chunks)

    answer = await asyncio.to_thread(_run)  # ★ 丢线程池
    ...

# agent/planner.py
async def execute_async(self, plan):
    ready = [t for t in plan if not t.depends_on]
    if ready:
        ready_results = await asyncio.gather(
            *(asyncio.to_thread(self._run_single, t) for t in ready)
        )
        ...
```

### 面试亮点
1. **不强求底层异步**：langchain 同步就让它同步，只在端点层用 `to_thread` 解放事件循环——这是改造存量项目最小破坏的做法。
2. **Planner 异步 + 同步双 API**：execute / execute_async 共存，调用方按场景选，单测都覆盖。
3. **为什么不全异步**：langchain 的 async API 在 tools 调用链上还不够稳定，强行 await 一圈引入更多 sharp edge。`to_thread` 是务实的中间方案。

### 可能被追问
- **Q：to_thread 用的默认线程池够吗？**
  A：默认 `min(32, cpu_count*5)`，对于 IO 密集的 LLM 调用基本够。要更高并发可以 `concurrent.futures.ThreadPoolExecutor(max_workers=N)` 显式建池然后 `loop.run_in_executor`。
- **Q：SSE 那段为什么还在用线程？**
  A：因为 LangChain `.stream()` 是同步生成器，包 to_thread 拿不到中间 chunk。所以专门起一个 producer thread 把 chunk 喂回 `queue.Queue`，主协程异步消费——这是把同步 generator 桥接到 async 流的标准模式。

---

## 9. 多级缓存（TTL+LRU / 语义 / 幂等 / Prefix）

### 痛点
Agent 应用最大的成本来源是 **同一问题反复调模型**、**同一工具被重复调用**。原来 `services/cache.py` 只是个字典壳子，没接入任何调用链。

### 实现要点
- **MemoryCache**：TTL + 容量上限的 LRU（基于 `OrderedDict` + `move_to_end`），线程安全。
- **SemanticCache**：基于 embedding 余弦相似度的近似查询缓存，保存完整 `RagResult`（answer/evidence/citations），并在 key 中隔离 tenant、知识库、语料、prompt、检索和模型版本；命中后行为与非缓存路径一致。
- **ToolCallCache**：工具调用幂等缓存。key = `sha1(tool_name + sorted(json(args)))`，TTL 默认 60s。
- **prefix hint**：`emit_prefix_cache_hint(prompt_prefix_chars)` 上报 system prompt 在前缀的字符数。豆包/通义都支持 prefix caching，系统提示词不变且固定在最前 → 服务端 KV cache 命中。
- **接入点**：
  - `RagSummarizeService.rag_summarize` 头部走 SemanticCache。
  - `agent/tools/middleware.py` 入口走 ToolCallCache，命中直接返回 `ToolMessage`，跳过工具与熔断。
- 所有命中/未命中都 `metrics_registry.inc_counter("agent_cache_hit_total" | "agent_cache_miss_total", {"cache": name})`。

### 关键代码

```python
# services/cache.py - LRU + TTL
def get(self, key):
    now = time.time()
    entry = self._store.get(key)
    if entry is None: return None
    expires_at, value = entry
    if expires_at < now:
        self._store.pop(key, None); return None
    self._store.move_to_end(key)
    return value

# services/cache.py - 语义命中
def get(self, query):
    vec = self._embed(query)
    for key in self._keys:
        cached_vec, value = self._memory.get(key)
        score = _cosine(vec, cached_vec)
        if score > best_score:
            best_score, best_value = score, value
    if best_score >= self.threshold:
        _record_hit(self.name)
        return best_value

# agent/tools/middleware.py - 幂等命中跳过工具
cached = tool_call_cache.get(tool_name, cache_args)
if cached is not None:
    metrics_registry.inc_tool_call(tool_name, status="cache_hit")
    event_bus.publish(request_id, "tool_end", {"status": "cache_hit", ...})
    return cached
```

### 面试亮点
1. **三层不同语义**：精确 key（MemoryCache）、近似 key（SemanticCache）、参数 hash key（ToolCallCache），覆盖不同场景。
2. **cache_hit 进 metrics + SSE 事件**：能向面试官展示"命中率 30% 时 P95 延迟从 1.2s 降到 0.4s"这种数据故事。
3. **prefix caching 是真实生产做法**：豆包/通义都有该能力，工程上只要保证 system prompt 在最前且不变，服务端就会复用 KV。我做的事是上报这个事实给 metrics 留证据，不需要改协议。

### 可能被追问
- **Q：语义缓存阈值怎么定？**
  A：embedding 模型不同上限不同。我跑过：`text-embedding-v4` 中文 query 上 0.92 是相对稳的，0.95 偏严格（漏命中多）、0.88 偏宽松（误命中多）。线上应该按场景做 A/B 然后定。
- **Q：工具幂等 TTL 60s 会不会太短？**
  A：`get_weather` 这种 60s 内确实可缓存；`fetch_external_data` 是用户数据，缓存可能拿到旧月份。所以 `ToolCallCache.register_ttl(tool_name, ttl)` 留了按工具单独配置 TTL 的口子，敏感工具设 5s 甚至 0。
- **Q：缓存击穿/雪崩怎么办？**
  A：演示项目暂没做，生产会加两层：
  1. 热点 key 永不全过期，后台异步刷新（lazy refresh）。
  2. miss 时加 single-flight（同一 key 同时只跑一次后端调用）。
  这两件在引入 Redis 后做更合适，当前内存版先满足主路径。

---

## 整体收口（结束语模板）

> "这一轮升级我的判断是：Agent 应用岗的工程深度，不在于 Prompt 写得多花，而在于 **请求进来怎么治理**——多模型路由、熔断降级、可观测三件套、多租户、评测体系、缓存策略，这六件事是字节豆包/Coze 这类平台的真实日常。我把它们都做到了能跑、能测、能讲、能改进。"
>
> "如果给我两周时间继续：第一周我会把 SQLite 替换成 Postgres 并上 Redis 限流；第二周接 OpenTelemetry SDK 真上 Tempo，并把语义缓存的命中率做 A/B 实验。"

---

## 附：核心测试覆盖一览

```
tests/test_circuit_breaker.py         6 个用例   三态机/恢复/嵌套调用
tests/test_model_router.py            6 个用例   provider 默认/权重/降级/租户过滤
tests/test_structured_logging.py      3 个用例   JSON 输出/ContextVar/脱敏
tests/test_prompt_versioning.py       5 个用例   frontmatter 解析/版本注入 ctx
tests/test_persistence_tenant.py      3 个用例   SQLite tenant 隔离/协议编码
tests/test_event_bus.py                           序号/replay/close/取消/隔离
tests/test_agent_streaming.py                     token/心跳/终态/身份隔离/SSE 格式
tests/test_planner_async.py           2 个用例   asyncio.gather 并发/依赖序列
tests/test_cache.py                   6 个用例   TTL/LRU/语义命中/metrics
tests/test_eval_harness_mode.py                   离线 Harness 指标与门禁
```

最终用例数以 CI 的 `pytest tests -q` 输出为准；同时执行 Ruff、30 条检索回归和 62 条 Agent 离线门禁。
