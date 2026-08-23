import inspect
from pathlib import Path

from app import (
    NAVIGATION,
    QUICK_PROMPTS,
    ROUTING_REASON_LABELS,
    _answer_evidence_from_terminal,
    _audit_event,
    _evaluation_artifact_sections,
    _event_detail,
    _event_label,
    _format_answer_with_citations,
    _inject_theme,
    _merge_answer,
)


def test_merge_answer_replaces_provisional_content_for_final_answer():
    assert _merge_answer("旧的中间内容", {"delta": "最终回答", "replace": True}) == "最终回答"


def test_merge_answer_appends_regular_delta():
    assert _merge_answer("您好", {"delta": "，请问有什么可以帮助您？"}) == (
        "您好，请问有什么可以帮助您？"
    )


def test_primary_navigation_keeps_complex_tasks_inside_chat():
    assert list(NAVIGATION) == ["对话", "记忆", "审批", "人工评测", "诊断"]
    assert "planner" not in NAVIGATION.values()
    assert len(QUICK_PROMPTS) == 4
    assert all("租户" not in label and "ID" not in label for label, _ in QUICK_PROMPTS)


def test_event_label_translates_backend_events_for_customer_view():
    assert _event_label("tool_started", {"tool": "rag_summarize"}) == (
        "正在调用企业知识库"
    )
    assert _event_label("verification_started", {}) == "正在校验答案的完整性与可靠性"


def test_auto_routing_and_plan_events_are_visible_in_the_chat_audit():
    routing = _audit_event(
        {
            "id": "2",
            "event": "routing_completed",
            "data": {
                "execution_mode": "plan_execute",
                "complexity_score": 7,
                "reasons": ["cross_domain_request"],
            },
        }
    )
    plan_step = _audit_event(
        {
            "id": "4",
            "event": "plan_step_completed",
            "data": {
                "id": "t1",
                "kind": "report",
                "description": "读取使用报告",
                "status": "completed",
                "result": "报告读取完成",
            },
        }
    )

    assert routing["label"] == "已自动选择处理方式：先规划再分步执行"
    assert routing["detail"]["处理方式"] == "先规划再分步执行"
    assert routing["detail"]["判定依据"] == ["涉及多个信息域"]
    assert plan_step["label"] == "计划步骤完成：读取使用报告"
    assert plan_step["detail"]["步骤结果"] == "报告读取完成"


def test_runtime_routing_transition_is_visible_without_exposing_reasoning_text():
    event = _audit_event(
        {
            "id": "5",
            "event": "routing_transition",
            "data": {
                "from_mode": "react",
                "to_mode": "plan_execute",
                "decision_source": "runtime_feedback",
                "reasons": ["verification_retry_escalation"],
            },
        }
    )

    assert event["label"] == (
        "处理方式已动态调整：边分析边调用所需服务 → 先规划再分步执行"
    )
    assert event["detail"]["调整依据"] == [
        "上轮结果未通过校验，升级为规划执行"
    ]


def test_budget_degradation_is_visible_as_a_verified_partial_result():
    event = _audit_event(
        {
            "id": "6",
            "event": "execution_degraded",
            "data": {
                "status": "completed",
                "reason": "retry_token_budget_insufficient",
                "strategy": "verified_partial_result",
            },
        }
    )

    assert event["label"] == "预算不足，已返回通过校验的部分结果"
    assert event["detail"]["降级原因"] == "剩余 Token 不足以执行安全重试"
    assert event["detail"]["降级策略"] == "verified_partial_result"


def test_evidence_excerpt_degradation_is_not_reported_as_budget_exhaustion():
    event = _audit_event(
        {
            "id": "7",
            "event": "execution_degraded",
            "data": {
                "status": "completed",
                "reason": "generated_answer_verification_failed",
                "strategy": "verified_evidence_excerpt",
            },
        }
    )

    assert event["label"] == "生成回答未通过校验，已返回通过校验的证据摘录"
    assert "预算" not in event["label"]
    assert event["detail"]["降级原因"] == "生成回答未通过证据校验"


def test_knowledge_gap_events_explain_the_actual_knowledge_base_limitation():
    expectations = {
        "knowledge_no_results": "知识库中没有找到相关内容，暂时无法依据现有资料回答",
        "knowledge_irrelevant": "知识库检索到的内容与问题不相关，暂时无法依据现有资料回答",
        "evidence_insufficient_for_conclusion": (
            "知识库现有资料不足以支持明确结论，暂时无法可靠回答"
        ),
    }

    for reason, expected_label in expectations.items():
        event = _audit_event(
            {
                "id": reason,
                "event": "execution_degraded",
                "data": {
                    "status": "completed",
                    "reason": reason,
                    "strategy": "knowledge_gap",
                },
            }
        )

        assert event["label"] == expected_label
        assert event["detail"] == {
            "结果类型": "知识库资料不足",
            "说明": ROUTING_REASON_LABELS[reason],
        }
        visible_text = f"{event['label']} {event['detail']}"
        assert "预算不足" not in visible_text
        assert "回答未通过校验" not in visible_text
        assert "执行降级" not in visible_text


def test_required_rag_routing_reason_has_a_friendly_label():
    assert ROUTING_REASON_LABELS["required_rag_execution_guarantee"] == (
        "知识型问题需先完成知识库检索"
    )


def test_answer_citations_render_as_reused_superscripts_with_friendly_tooltips():
    evidence_id = "928c24dbdeeb1c0a109f5620c8bac429:40"
    answer = f"先清理风道 [{evidence_id}]，再检查吸力 [{evidence_id}]。"
    evidence = [
        {
            "id": evidence_id,
            "source": r"E:\knowledge\故障排除.txt",
            "excerpt": "风道有异响时，清理风道杂物并检查电机。",
            "chunk_index": 40,
        }
    ]

    rendered = _format_answer_with_citations(answer, evidence)

    assert evidence_id not in rendered
    assert rendered.count('class="citation-ref"') == 2
    assert rendered.count('<sup aria-hidden="true">1</sup>') == 2
    assert "故障排除.txt" in rendered
    assert "片段 40" in rendered
    assert "风道有异响时，清理风道杂物并检查电机。" in rendered


def test_answer_citations_escape_answer_and_evidence_html():
    rendered = _format_answer_with_citations(
        '<img src=x onerror="alert(1)"> [manual:7]',
        [
            {
                "id": "manual:7",
                "source": "<img onerror=bad>.txt",
                "content": "</span><script>alert(2)</script>",
            }
        ],
    )

    assert "<img" not in rendered
    assert "<script>" not in rendered
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in rendered
    assert "&lt;img onerror=bad&gt;.txt" in rendered
    assert "&lt;/span&gt;&lt;script&gt;alert(2)&lt;/script&gt;" in rendered


def test_unknown_citation_metadata_hides_raw_id_and_degrades_safely():
    evidence_id = "928c24dbdeeb1c0a109f5620c8bac429:40"

    rendered = _format_answer_with_citations(f"清理耗材 [{evidence_id}]", [])

    assert evidence_id not in rendered
    assert '<sup aria-hidden="true">1</sup>' in rendered
    assert "来源详情暂不可用" in rendered


def test_plain_bracketed_number_is_not_inferred_as_a_citation():
    rendered = _format_answer_with_citations("型号范围为 [2026]。", [])

    assert rendered == "型号范围为 [2026]。"
    assert "citation-ref" not in rendered


def test_run_audit_retains_raw_evidence_ids_outside_the_answer_body():
    detail = _event_detail(
        "run_completed",
        {
            "status": "completed",
            "evidence": [
                {
                    "id": "928c24dbdeeb1c0a109f5620c8bac429:40",
                    "source": "故障排除.txt",
                }
            ],
        },
    )

    assert detail["证据定位编号"] == ["928c24dbdeeb1c0a109f5620c8bac429:40"]


def test_terminal_payload_supplies_public_citation_metadata():
    evidence = [{"id": "manual:2", "source": "手册.pdf", "content": "清理滤网"}]
    payload = {"evidence": evidence}

    assert _answer_evidence_from_terminal(payload) == evidence


def test_terminal_answer_artifact_is_supported_for_old_servers():
    evidence = [{"id": "manual:2", "source": "手册.pdf", "content": "清理滤网"}]

    assert _answer_evidence_from_terminal(
        {
            "artifacts": [
                {
                    "artifact_type": "answer",
                    "name": "final-answer",
                    "payload": {"evidence": evidence},
                }
            ]
        }
    ) == evidence


def test_verified_rag_recovery_has_an_accurate_non_budget_label():
    label = _event_label(
        "execution_degraded",
        {
            "reason": "generated_answer_verification_failed",
            "strategy": "verified_rag_answer",
        },
    )

    assert label == "最终整合遗漏了引用，已恢复通过校验的知识库回答"
    assert "预算" not in label


def test_non_budget_degradation_uses_a_neutral_label():
    assert _event_label(
        "execution_degraded",
        {
            "reason": "generated_answer_verification_failed",
            "strategy": "stop_unverified_retry",
        },
    ) == "结果未通过校验，已停止输出未经校验的结果"


def test_documented_budget_defaults_are_consistent():
    project_root = Path(__file__).resolve().parents[1]
    env_values = {
        key: value
        for line in (project_root / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
        for key, value in [line.split("=", 1)]
    }
    expected = {
        "AGENT_MAX_RUN_TOKENS": "32000",
        "AGENT_MAX_MODEL_OUTPUT_TOKENS": "2000",
        "AGENT_MODEL_MAX_RETRIES": "2",
        "AGENT_MAX_TOOL_CALLS": "8",
        "AGENT_MAX_STEPS": "8",
        "AGENT_MAX_COST": "1.0",
        "AGENT_MAX_VERIFICATION_RETRIES": "1",
        "AGENT_MIN_REPAIR_TOKENS": "4500",
        "AGENT_RAG_MAX_OUTPUT_TOKENS": "1600",
        "AGENT_MAX_REACT_RECURSION": "12",
        "AGENT_MEMORY_EXTRACTION_MAX_TOKENS": "900",
        "AGENT_SUMMARY_MAX_TOKENS": "500",
    }

    assert {key: env_values[key] for key in expected} == expected

    compose = (project_root / "docker-compose.yml").read_text(encoding="utf-8")
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    for key, value in expected.items():
        assert f"{key}: ${{{key}:-{value}}}" in compose
        assert f"| `{key}` | `{value}` |" in readme


def test_audit_event_keeps_structured_results_and_drops_stream_noise():
    event = _audit_event(
        {
            "id": "4",
            "event": "tool_completed",
            "data": {
                "tool": "rag_summarize",
                "status": "success",
                "duration_ms": 18.2,
                "result": "滤网应定期清理",
                "result_truncated": False,
            },
        }
    )

    assert event["id"] == "4"
    assert event["detail"]["工具"] == "企业知识库"
    assert event["detail"]["返回结果"] == "滤网应定期清理"
    assert _audit_event({"event": "token_delta", "data": {"delta": "x"}}) is None
    assert _audit_event({"event": "heartbeat", "data": {}}) is None


def test_hard_token_limit_degradation_is_reported_as_budget_limited():
    assert _event_label(
        "execution_degraded",
        {
            "reason": "max_tokens_exceeded",
            "strategy": "stop_unverified_retry",
        },
    ) == "预算不足，已停止输出未经校验的结果"


def test_terminal_event_shows_actual_budget_usage():
    detail = _event_detail(
        "run_failed",
        {
            "status": "blocked",
            "error": "max_tokens_exceeded",
            "budget": {
                "used_tokens": 32000,
                "max_tokens": 32000,
                "used_tool_calls": 3,
                "max_tool_calls": 8,
            },
        },
    )

    assert detail["Token 用量"] == "32000 / 32000"
    assert detail["工具调用用量"] == "3 / 8"


def test_verification_detail_exposes_quality_result():
    detail = _event_detail(
        "verification_completed",
        {
            "passed": False,
            "action": "retry",
            "score": 0.6,
            "reasons": ["citation_missing"],
            "citation_coverage": 0.5,
        },
    )

    assert detail["是否通过"] is False
    assert detail["未通过原因"] == ["citation_missing"]
    assert detail["引用覆盖率"] == 0.5


def test_human_eval_artifacts_separate_approvals_plans_and_actual_tools():
    payload = {
        "approval_records": [{"approval_id": "approval-1", "status": "approved"}],
        "planner_steps": [{"id": "t1", "kind": "weather", "city": "杭州"}],
        "tool_calls": [
            {"tool_name": "get_weather", "args": {"city": "杭州"}, "status": "success"}
        ],
        "trace": [
            {"category": "planner", "name": "plan"},
            {"category": "tool", "name": "get_weather"},
            {"category": "diagnostic", "name": "verifier"},
        ],
    }

    sections = _evaluation_artifact_sections(payload)

    assert sections["approval_records"][0]["approval_id"] == "approval-1"
    assert sections["planner_steps"][0]["id"] == "t1"
    assert sections["tool_calls"][0]["tool_name"] == "get_weather"
    assert sections["other_trace"] == [
        {"category": "diagnostic", "name": "verifier"}
    ]


def test_human_eval_artifacts_keep_legacy_trace_visible_in_separate_sections():
    sections = _evaluation_artifact_sections(
        {
            "trace": [
                {
                    "category": "planner",
                    "name": "plan",
                    "metadata": {"task_count": 2},
                },
                {
                    "category": "tool",
                    "name": "get_weather",
                    "metadata": {
                        "redacted_args": {"city": "杭州"},
                        "result": "杭州晴",
                    },
                },
            ]
        }
    )

    assert sections["planner_steps"][0]["event"] == "plan"
    assert sections["tool_calls"] == [
        {
            "tool_name": "get_weather",
            "args": {"city": "杭州"},
            "status": "success",
            "result": "杭州晴",
        }
    ]


def test_sidebar_controls_remain_available_after_collapse():
    theme_source = inspect.getsource(_inject_theme)

    assert '[data-testid="stToolbar"] { display: flex !important; }' in theme_source
    assert '[data-testid="stSidebarCollapseButton"] {' in theme_source
    assert '[data-testid="stExpandSidebarButton"] {' in theme_source
    assert (
        '[data-testid="stToolbar"], #MainMenu, footer { display: none; }'
        not in theme_source
    )
