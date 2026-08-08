from __future__ import annotations

import html
import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from services.frontend_api import AgentApiClient, AgentApiError


MEMORY_CATEGORIES = {
    "设备型号/归属": "device_identity",
    "设备状态": "device_state",
    "用户偏好": "user_preference",
    "稳定用户信息": "stable_profile",
    "未完成事项": "open_item",
    "安全/权限约束": "safety_constraint",
    "用户响应策略": "user_policy",
    "普通情景": "episodic",
    "临时状态": "transient",
}

NAVIGATION = {
    "对话": "chat",
    "记忆": "memory",
    "审批": "operations",
    "诊断": "observability",
}

EXECUTION_MODE_LABELS = {
    "direct": "直接回答",
    "react": "边分析边调用所需服务",
    "plan_execute": "先规划再分步执行",
}

ROUTING_REASON_LABELS = {
    "conversational_request": "简短对话",
    "single_objective": "单一目标",
    "tool_or_knowledge_request": "需要工具或知识支持",
    "cross_domain_request": "涉及多个信息域",
    "explicit_multi_step_request": "用户明确要求分步处理",
    "multiple_actions": "包含多个处理动作",
    "multiple_objectives": "包含多个交付目标",
    "long_context_request": "请求上下文较长",
    "router_error_fallback": "自动判断异常，已安全回退到常规 Agent",
    "semantic_single_goal": "语义判断为单一目标",
    "semantic_multiple_goals": "语义判断为多个目标",
    "semantic_tool_required": "目标需要服务或工具支持",
    "semantic_dependencies": "目标之间存在执行依赖",
    "semantic_context_followup": "当前请求依赖最近对话",
    "semantic_high_risk": "任务包含较高风险操作",
    "semantic_consistency_upgrade": "结构化目标需要升级执行方式",
    "single_goal_plan_downgrade": "单一目标无需完整规划",
    "low_semantic_confidence": "语义判断置信度不足，回退到常规 Agent",
    "required_capability_unavailable": "所需能力当前不可用",
    "plan_step_budget_insufficient": "剩余步骤预算不足",
    "plan_tool_budget_insufficient": "剩余工具预算不足",
    "plan_token_budget_insufficient": "剩余 Token 预算不足",
    "tool_requirement_forces_react": "任务需要工具，已切换到常规 Agent",
    "semantic_router_fallback": "语义路由不可用，已使用确定性兜底",
    "verification_retry_escalation": "上轮结果未通过校验，升级为规划执行",
    "planner_verification_retry_downgrade": "规划结果未通过校验，降级为常规 Agent 重试",
    "semantic_report_capability_required": "语义目标需要读取报告数据",
    "verification_retry_budget_insufficient": "剩余预算不足以执行安全重试",
    "retry_token_budget_insufficient": "剩余 Token 不足以执行安全重试",
    "retry_tool_budget_insufficient": "剩余工具额度不足以执行安全重试",
}

TOOL_LABELS = {
    "rag_summarize": "企业知识库",
    "get_weather": "环境信息服务",
    "get_user_location": "客户位置服务",
    "get_user_id": "客户身份服务",
    "get_current_month": "时间服务",
    "fetch_external_data": "客户数据服务",
    "fill_context_for_report": "报告上下文服务",
}

QUICK_PROMPTS = (
    ("帮我排查清洁效果下降", "扫地机器人最近清洁效果下降，应该如何系统排查？"),
    ("制定耗材维护计划", "请告诉我滤网、边刷和主刷的日常维护建议。"),
    ("结合家庭环境给选购建议", "请根据我的家庭环境，帮我分析适合哪类扫地机器人。"),
    ("查看设备使用注意事项", "请总结扫地机器人日常使用中最重要的安全和维护注意事项。"),
)


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-primary: #2f2f2f;
            --text-secondary: #6f6f6f;
            --text-tertiary: #8f8f8f;
            --line: #e8e8e8;
            --line-strong: #d7d7d7;
            --surface: #ffffff;
            --surface-soft: #f7f7f8;
            --surface-hover: #ececec;
            --accent: #10a37f;
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                "PingFang SC", "Microsoft YaHei", sans-serif;
            color: var(--text-primary);
        }

        .stApp { background: var(--surface); }
        [data-testid="stHeader"] {
            background: rgba(255,255,255,.92);
            height: 2.75rem;
        }
        /*
         * Keep Streamlit's toolbar layout alive: the "expand sidebar" control is
         * mounted inside it after the sidebar is collapsed. Hiding the whole
         * toolbar makes collapsing a one-way action.
         */
        [data-testid="stToolbar"] { display: flex !important; }
        [data-testid="stToolbarActions"],
        [data-testid="stAppDeployButton"],
        #MainMenu,
        footer {
            display: none !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stExpandSidebarButton"] {
            width: 2rem;
            height: 2rem;
            border-radius: 9px;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stExpandSidebarButton"]:hover {
            background: var(--surface-hover) !important;
        }
        [data-testid="stExpandSidebarButton"] {
            margin: .35rem;
            background: rgba(255,255,255,.94) !important;
            border: 1px solid var(--line) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,.08);
        }
        .block-container {
            max-width: 1180px;
            padding-top: .4rem;
            padding-bottom: 7rem;
        }
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span {
            color: var(--text-primary);
        }

        section[data-testid="stSidebar"] {
            background: var(--surface-soft);
            border-right: 1px solid #ededed;
            width: 260px !important;
            min-width: 260px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: .65rem .65rem 1rem;
        }
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] summary {
            color: var(--text-primary) !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            display: none;
        }
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
            gap: .15rem;
        }
        section[data-testid="stSidebar"] .stRadio [role="radio"] {
            display: none;
        }
        section[data-testid="stSidebar"] .stRadio label {
            min-height: 38px;
            width: 100%;
            padding: .35rem .6rem;
            border-radius: 9px;
            transition: background .15s ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
            color: var(--text-primary) !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: var(--surface-hover);
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: #e7e7e8;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 0;
            background: transparent;
        }
        section[data-testid="stSidebar"] .stButton button {
            color: var(--text-primary);
            background: transparent;
            border: 1px solid transparent;
            border-radius: 9px;
            justify-content: flex-start;
            font-weight: 500;
        }
        section[data-testid="stSidebar"] .stButton button p {
            width: 100%;
            color: var(--text-primary) !important;
            text-align: left;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: var(--surface-hover);
            border-color: transparent;
        }
        section[data-testid="stSidebar"] input {
            color: var(--text-primary) !important;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: .65rem;
            padding: .25rem .45rem .7rem;
        }
        .brand-mark {
            width: 30px;
            height: 30px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            color: white;
            font-size: .8rem;
            background: #111111;
        }
        .brand-name {
            color: var(--text-primary);
            font-weight: 650;
            font-size: .94rem;
        }
        .brand-subtitle { display: none; }
        .sidebar-section-label {
            color: var(--text-tertiary);
            font-size: .7rem;
            font-weight: 600;
            padding: .8rem .6rem .35rem;
        }
        .sidebar-status {
            display: flex;
            align-items: center;
            gap: .45rem;
            color: var(--text-secondary);
            font-size: .74rem;
            padding: .55rem .65rem .15rem;
        }
        .sidebar-status.offline { color: #b42318; }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent);
        }
        .sidebar-status.offline .status-dot { background: #ef4444; }

        .page-heading {
            max-width: 860px;
            margin: 1.25rem auto 1.5rem;
        }
        .page-kicker {
            color: var(--text-tertiary);
            font-size: .72rem;
            font-weight: 600;
        }
        .page-heading h1 {
            color: var(--text-primary);
            font-size: 1.65rem;
            line-height: 1.25;
            margin: .2rem 0 .45rem;
            letter-spacing: -.02em;
        }
        .page-heading p {
            color: var(--text-secondary);
            font-size: .9rem;
            margin: 0;
            max-width: 720px;
        }

        .chat-empty {
            text-align: center;
            padding: 12vh 1rem 2.2rem;
        }
        .chat-empty-mark {
            width: 42px;
            height: 42px;
            margin: 0 auto 1.1rem;
            display: grid;
            place-items: center;
            color: white;
            background: #111;
            border-radius: 50%;
            font-size: 1rem;
        }
        .chat-empty h1 {
            color: var(--text-primary);
            font-size: 1.9rem;
            font-weight: 600;
            letter-spacing: -.035em;
            margin: 0 0 .55rem;
        }
        .chat-empty p {
            color: var(--text-secondary);
            font-size: .9rem;
            margin: 0 auto;
            max-width: 560px;
            line-height: 1.65;
        }

        div[data-testid="stChatMessage"] {
            max-width: 850px;
            margin: 0 auto;
            background: transparent;
            border: 0;
            box-shadow: none;
            padding: .65rem .2rem;
        }
        div[data-testid="stChatMessage"] p {
            color: var(--text-primary);
            line-height: 1.75;
        }
        [data-testid="stChatInput"] {
            max-width: 850px;
            margin: 0 auto 1rem;
            border-radius: 26px;
            border: 1px solid var(--line-strong);
            background: white;
            box-shadow: 0 5px 24px rgba(0,0,0,.08);
            overflow: hidden;
        }
        [data-testid="stBottomBlockContainer"] {
            background: rgba(255,255,255,.96) !important;
            border-top: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"] {
            background: white !important;
            background-color: white !important;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }
        [data-testid="stChatInput"] button {
            border-radius: 50% !important;
            background: #111 !important;
            color: white !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border-color: var(--line) !important;
            border-radius: 14px;
            box-shadow: none;
        }
        .stButton button {
            min-height: 40px;
            border-radius: 12px;
            border: 1px solid var(--line);
            color: var(--text-primary);
            background: white;
            font-weight: 500;
        }
        .stButton button:hover {
            border-color: var(--line-strong);
            background: var(--surface-soft);
            color: var(--text-primary);
        }
        .stButton button[kind="primary"] {
            color: white;
            background: #111;
            border-color: #111;
        }
        .stButton button[kind="primary"] p {
            color: white !important;
        }
        div[data-testid="stMetric"] {
            padding: .75rem .85rem;
            border-radius: 12px;
            background: white;
            border: 1px solid var(--line);
        }
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            opacity: 1 !important;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .stTabs [data-baseweb="tab-list"] {
            gap: .25rem;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            padding: .5rem .75rem;
            color: var(--text-secondary) !important;
            opacity: 1 !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: var(--text-secondary) !important;
            opacity: 1 !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text-primary);
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] p {
            color: var(--text-primary) !important;
        }
        div[data-testid="stStatusWidget"] {
            max-width: 850px;
            margin: .35rem auto;
            border-color: var(--line) !important;
            box-shadow: none !important;
        }
        @media (max-width: 800px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .chat-empty { padding-top: 7vh; }
            .chat-empty h1 { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    configured_api_url = os.getenv("AGENT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    defaults = {
        "chat_messages": [],
        "session_id": str(uuid4()),
        "last_request_id": "",
        "approval_id": "",
        "pending_approval_request": {},
        "queued_prompt": "",
        "queued_approval_id": "",
        "backend_online": False,
        "connection_error": "",
        "navigation": "对话",
        "pending_navigation": "",
        "api_key": os.getenv("AGENT_API_KEY", "dev-api-key"),
        "bearer_token": os.getenv("AGENT_UI_BEARER_TOKEN", ""),
        "tenant_id": os.getenv("AGENT_UI_TENANT_ID", "tenant-a"),
        "user_id": os.getenv("AGENT_UI_USER_ID", "user-1005"),
        "user_role": os.getenv("AGENT_UI_USER_ROLE", "user"),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    # The deployment controls the backend endpoint. Never retain a stale URL in a browser session.
    st.session_state.api_url = configured_api_url


def _merge_answer(answer: str, payload: dict) -> str:
    delta = str(payload.get("delta", ""))
    return delta if payload.get("replace") else answer + delta


def _event_label(event_type: str, payload: dict) -> str:
    tool_name = payload.get("tool") or payload.get("tool_name") or "服务工具"
    tool_label = TOOL_LABELS.get(str(tool_name), str(tool_name))
    if event_type == "routing_completed":
        mode = EXECUTION_MODE_LABELS.get(
            str(payload.get("execution_mode")),
            str(payload.get("execution_mode") or "常规处理"),
        )
        return f"已自动选择处理方式：{mode}"
    if event_type == "routing_transition":
        from_mode = EXECUTION_MODE_LABELS.get(
            str(payload.get("from_mode")),
            str(payload.get("from_mode") or "原处理方式"),
        )
        to_mode = EXECUTION_MODE_LABELS.get(
            str(payload.get("to_mode")),
            str(payload.get("to_mode") or "新处理方式"),
        )
        return f"处理方式已动态调整：{from_mode} → {to_mode}"
    if event_type == "plan_created":
        return f"已生成 {len(payload.get('steps') or [])} 步执行计划"
    if event_type == "plan_step_started":
        return f"正在执行：{payload.get('description') or payload.get('id') or '计划步骤'}"
    if event_type == "plan_step_completed":
        status = "完成" if payload.get("status") == "completed" else "未完成"
        return f"计划步骤{status}：{payload.get('description') or payload.get('id') or ''}"
    if event_type == "plan_completed":
        if payload.get("status") == "partial":
            return "复杂任务已完成可执行部分，部分步骤未完成"
        return "复杂任务已完成规划与执行"
    if event_type == "execution_degraded":
        if payload.get("strategy") == "verified_partial_result":
            return "预算不足，已返回通过校验的部分结果"
        return "预算不足，已停止输出未经校验的结果"
    labels = {
        "run_started": "已接收问题，正在理解服务需求",
        "model_started": "正在分析问题并制定处理路径",
        "model_completed": "本轮分析与信息收集完成",
        "execution_context_updated": "已根据目标更新受控执行场景",
        "tool_started": f"正在调用{tool_label}",
        "tool_completed": f"{tool_label}返回完成",
        "tool_skipped": f"已跳过重复或非必要的{tool_label}调用",
        "tool_failed": f"{tool_label}暂时不可用",
        "verification_started": "正在校验答案的完整性与可靠性",
        "verification_completed": "答案质量校验完成",
        "approval_required": "本步骤需要人工审批",
        "artifact_created": "已生成可追踪的服务产物",
        "memory_operation_completed": "记忆操作已完成并确认落库",
        "heartbeat": "服务仍在处理中，请稍候",
        "run_completed": "服务处理完成",
        "run_failed": "本次服务未能完成",
    }
    return labels.get(event_type, event_type.replace("_", " "))


def _event_detail(event_type: str, payload: dict) -> dict:
    """Build a bounded audit view without exposing hidden chain-of-thought."""
    tool_name = str(payload.get("tool") or "")
    tool_label = TOOL_LABELS.get(tool_name, tool_name)
    if event_type == "execution_degraded":
        reason = str(payload.get("reason") or "")
        return {
            "执行状态": payload.get("status"),
            "降级原因": ROUTING_REASON_LABELS.get(reason, reason),
            "降级策略": payload.get("strategy"),
        }
    mappings = {
        "run_started": {
            "执行场景": payload.get("scene"),
            "会话编号": payload.get("session_id"),
            "最大步骤数": payload.get("max_steps"),
            "最大工具调用数": payload.get("max_tool_calls"),
            "Token 总预算": payload.get("max_tokens"),
        },
        "model_started": {
            "分析轮次": int(payload.get("attempt", 0)) + 1,
            "本轮最大输出 Token": payload.get("max_output_tokens"),
        },
        "model_completed": {
            "分析轮次": int(payload.get("attempt", 0)) + 1,
            "模型": payload.get("model_name"),
            "输入 Token": payload.get("tokens_in"),
            "输出 Token": payload.get("tokens_out"),
            "剩余 Token": payload.get("remaining_tokens"),
            "工具结果数": payload.get("tool_result_count"),
            "证据数": payload.get("evidence_count"),
            "成本": payload.get("cost"),
            "成本口径": payload.get("cost_mode"),
        },
        "execution_context_updated": {
            "执行场景": payload.get("scene"),
            "更新原因": ROUTING_REASON_LABELS.get(
                str(payload.get("reason")),
                payload.get("reason"),
            ),
        },
        "routing_completed": {
            "处理方式": EXECUTION_MODE_LABELS.get(
                str(payload.get("execution_mode")),
                payload.get("execution_mode"),
            ),
            "复杂度评分": payload.get("complexity_score"),
            "语义置信度": payload.get("confidence"),
            "决策来源": payload.get("decision_source"),
            "任务风险": payload.get("risk"),
            "结构化目标": payload.get("goals") or [],
            "所需工具": payload.get("required_tools") or [],
            "不可用工具": payload.get("unavailable_tools") or [],
            "动态调整": payload.get("transition"),
            "判定依据": [
                ROUTING_REASON_LABELS.get(str(reason), str(reason))
                for reason in (payload.get("reasons") or [])
            ],
            "路由版本": payload.get("router_version"),
        },
        "routing_transition": {
            "原处理方式": EXECUTION_MODE_LABELS.get(
                str(payload.get("from_mode")),
                payload.get("from_mode"),
            ),
            "新处理方式": EXECUTION_MODE_LABELS.get(
                str(payload.get("to_mode")),
                payload.get("to_mode"),
            ),
            "调整来源": payload.get("decision_source"),
            "调整依据": [
                ROUTING_REASON_LABELS.get(str(reason), str(reason))
                for reason in (payload.get("reasons") or [])
            ],
        },
        "plan_created": {
            "执行计划": payload.get("steps") or [],
            "步骤数量": len(payload.get("steps") or []),
        },
        "plan_step_started": {
            "步骤编号": payload.get("id"),
            "步骤类型": payload.get("kind"),
            "步骤目标": payload.get("description"),
        },
        "plan_step_completed": {
            "步骤编号": payload.get("id"),
            "步骤类型": payload.get("kind"),
            "步骤目标": payload.get("description"),
            "执行状态": payload.get("status"),
            "步骤结果": payload.get("result"),
            "失败原因": payload.get("error"),
        },
        "plan_completed": {
            "执行状态": payload.get("status"),
            "计划步骤数": payload.get("step_count"),
            "成功步骤数": payload.get("successful_steps"),
        },
        "tool_started": {
            "工具": tool_label,
            "调用参数": payload.get("args") or {},
        },
        "tool_completed": {
            "工具": tool_label,
            "状态": payload.get("status"),
            "耗时（毫秒）": payload.get("duration_ms"),
            "返回结果": payload.get("result", "未提供结果摘要"),
            "结果是否截断": bool(payload.get("result_truncated")),
        },
        "tool_skipped": {
            "工具": tool_label,
            "状态": payload.get("status"),
            "跳过原因": payload.get("result") or payload.get("reason") or "重复或非必要调用",
        },
        "tool_failed": {
            "工具": tool_label,
            "状态": payload.get("status"),
            "失败原因": payload.get("error") or payload.get("reason") or "工具暂时不可用",
            "耗时（毫秒）": payload.get("duration_ms"),
        },
        "verification_started": {
            "校验轮次": int(payload.get("attempt", 0)) + 1,
            "参与校验的证据数": payload.get("evidence_count"),
        },
        "verification_completed": {
            "是否通过": payload.get("passed"),
            "后续动作": payload.get("action"),
            "综合分数": payload.get("score"),
            "未通过原因": payload.get("reasons") or [],
            "引用有效率": payload.get("citation_validity"),
            "引用覆盖率": payload.get("citation_coverage"),
            "无证据声明率": payload.get("unsupported_claim_rate"),
            "是否含有害指令": payload.get("harmful_instruction"),
        },
        "approval_required": {
            "工具": tool_label,
            "审批编号": payload.get("approval_id"),
        },
        "artifact_created": {
            "产物编号": payload.get("artifact_id"),
            "产物类型": payload.get("artifact_type") or payload.get("type"),
            "产物名称": payload.get("name"),
            "生成时间": payload.get("created_at"),
        },
        "memory_operation_completed": {
            "操作": payload.get("action"),
            "写入状态": payload.get("status"),
            "记忆编号": payload.get("memory_ids") or [],
            "记忆键": payload.get("saved_keys") or [],
            "删除数量": payload.get("deleted"),
            "拒绝原因": payload.get("rejected_reason") or "",
        },
        "run_completed": {
            "最终状态": payload.get("status"),
            "产物数量": len(payload.get("artifacts") or []),
            "是否经过质量校验": bool(payload.get("verifier")),
        },
        "run_failed": {
            "最终状态": payload.get("status"),
            "失败原因": payload.get("error") or "执行未完成",
        },
    }
    return {key: value for key, value in mappings.get(event_type, {}).items() if value is not None}


def _audit_event(event: dict) -> dict | None:
    event_type = str(event.get("event") or "")
    if event_type in {"token_delta", "heartbeat"}:
        return None
    payload = event.get("data") if isinstance(event.get("data"), dict) else {}
    return {
        "id": str(event.get("id") or ""),
        "event": event_type,
        "label": _event_label(event_type, payload),
        "detail": _event_detail(event_type, payload),
    }


def _render_audit_trail(events: list[dict], request_id: str) -> None:
    if not events:
        return
    with st.expander("查看本次服务过程与每步结果"):
        st.caption(f"服务请求编号 · {request_id}")
        st.info("这里展示可审计的模型决策、工具调用和质量校验结果，不展示隐藏的内部思维文本。")
        for index, event in enumerate(events, start=1):
            sequence = event.get("id") or str(index)
            st.markdown(f"**{index:02d} · {event['label']}**  `事件 {sequence}`")
            detail = event.get("detail") or {"说明": "该步骤没有额外的结构化结果。"}
            st.json(detail, expanded=True)


def _brand_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">✦</div>
          <div>
            <div class="brand-name">智能客服</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _start_new_chat() -> None:
    st.session_state.session_id = str(uuid4())
    st.session_state.chat_messages = []
    st.session_state.last_request_id = ""
    st.session_state.approval_id = ""
    st.session_state.pending_approval_request = {}
    st.session_state.queued_approval_id = ""
    st.session_state.navigation = "对话"


def _sidebar_client() -> tuple[str, AgentApiClient] | None:
    _brand_sidebar()
    if st.sidebar.button("＋  新对话", use_container_width=True):
        _start_new_chat()
        st.rerun()

    pending_navigation = st.session_state.pop("pending_navigation", "")
    if pending_navigation:
        st.session_state.navigation = pending_navigation
    if st.session_state.navigation not in NAVIGATION:
        st.session_state.navigation = "对话"
    page_label = st.sidebar.radio(
        "导航",
        list(NAVIGATION),
        label_visibility="collapsed",
        key="navigation",
    )

    missing = ["服务地址"] if not str(st.session_state.api_url).strip() else []
    has_credential = bool(
        str(st.session_state.api_key).strip()
        or str(st.session_state.bearer_token).strip()
    )
    client = None
    if not missing and has_credential:
        client = AgentApiClient(
            st.session_state.api_url,
            st.session_state.api_key,
            st.session_state.tenant_id,
            st.session_state.user_id,
            user_role=st.session_state.user_role,
            bearer_token=st.session_state.bearer_token,
        )
        try:
            client.health()
            identity = client.identity()
            st.session_state.tenant_id = str(identity.get("tenant_id") or "")
            st.session_state.user_id = str(identity.get("user_id") or "")
            st.session_state.user_role = str(identity.get("role") or "user")
            st.session_state.backend_online = True
            st.session_state.connection_error = ""
        except AgentApiError as exc:
            st.session_state.backend_online = False
            st.session_state.connection_error = str(exc)
    else:
        st.session_state.backend_online = False

    if client is not None and st.session_state.backend_online:
        try:
            sessions = client.list_sessions(limit=8)
        except AgentApiError:
            sessions = []
        if sessions:
            st.sidebar.markdown(
                '<div class="sidebar-section-label">最近对话</div>',
                unsafe_allow_html=True,
            )
            for item in sessions:
                session_id = str(item.get("session_id") or "")
                title = str(item.get("title") or "未命名对话").strip()
                label = title if len(title) <= 22 else title[:22].rstrip() + "…"
                if st.sidebar.button(
                    label,
                    key=f"recent_session_{session_id}",
                    use_container_width=True,
                ):
                    try:
                        payload = client.session_messages(session_id)
                        st.session_state.session_id = session_id
                        st.session_state.chat_messages = payload.get("messages", [])
                        st.session_state.last_request_id = ""
                        st.session_state.navigation = "对话"
                        st.rerun()
                    except AgentApiError as exc:
                        st.sidebar.error(f"恢复失败：{exc}")

    status_text = "服务正常" if st.session_state.backend_online else "连接异常"
    status_class = (
        "sidebar-status"
        if st.session_state.backend_online
        else "sidebar-status offline"
    )
    st.sidebar.markdown(
        f"""
        <div class="{status_class}">
          <span class="status-dot"></span>{status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("连接与诊断", expanded=False):
        st.caption("以下信息仅用于部署、联调与故障排查。")
        st.text_input("服务地址", key="api_url", disabled=True)
        st.text_input("访问密钥", key="api_key", type="password")
        st.text_input("Bearer Token", key="bearer_token", type="password")
        if st.session_state.backend_online:
            st.caption(
                f"已认证身份 · {st.session_state.user_id} · {st.session_state.user_role}"
            )
        elif st.session_state.connection_error:
            st.error(f"认证失败：{st.session_state.connection_error}")
        st.caption(f"当前会话 · {st.session_state.session_id[:8]}")

    if missing:
        st.sidebar.error("请在连接与诊断中补充：" + "、".join(missing))
        return None
    if not has_credential:
        st.sidebar.error("请在连接与诊断中补充访问凭据")
        return None
    if client is None:
        return None
    return NAVIGATION[page_label], client


def _page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
          <div class="page-kicker">{html.escape(kicker)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _queue_prompt(prompt: str) -> None:
    st.session_state.queued_prompt = prompt
    st.rerun()


def _render_chat(client: AgentApiClient) -> None:
    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div class="chat-empty">
              <div class="chat-empty-mark">✦</div>
              <h1>有什么可以帮你？</h1>
              <p>描述问题或想完成的目标。我会自动判断是直接回答、调用所需服务，
              还是先制定计划再分步执行。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(QUICK_PROMPTS), 2):
            suggestion_columns = st.columns(2, gap="small")
            for column, (label, prompt) in zip(
                suggestion_columns,
                QUICK_PROMPTS[row_start : row_start + 2],
            ):
                if column.button(
                    label,
                    key=f"welcome_{label}",
                    use_container_width=True,
                ):
                    _queue_prompt(prompt)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("events"):
                _render_audit_trail(
                    message["events"], message.get("request_id", "")
                )

    queued_prompt = str(st.session_state.pop("queued_prompt", "") or "")
    queued_approval_id = str(
        st.session_state.pop("queued_approval_id", "") or ""
    )
    typed_prompt = st.chat_input("向智能客服提问")
    prompt = queued_prompt or typed_prompt
    if prompt:
        _process_chat_request(
            client,
            prompt,
            approval_id=queued_approval_id or None,
            append_user=not bool(queued_approval_id),
        )


def _process_chat_request(
    client: AgentApiClient,
    prompt: str,
    *,
    approval_id: str | None = None,
    append_user: bool = True,
) -> None:
    if append_user:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    request_id = str(uuid4())
    st.session_state.last_request_id = request_id
    answer = ""
    terminal_payload = {}
    event_log = []

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status = st.status("正在受理您的服务请求", expanded=True)
        try:
            for event in client.chat_events(
                prompt,
                st.session_state.session_id,
                request_id=request_id,
                approval_id=approval_id,
            ):
                event_type = event["event"]
                payload = event["data"]
                audit_event = _audit_event(event)
                if audit_event is not None:
                    event_log.append(audit_event)
                label = _event_label(event_type, payload)
                if event_type == "token_delta":
                    answer = _merge_answer(answer, payload)
                    answer_placeholder.markdown(answer + "▌")
                elif event_type in {
                    "run_started",
                    "model_started",
                    "model_completed",
                    "execution_context_updated",
                    "execution_degraded",
                    "routing_completed",
                    "routing_transition",
                    "plan_created",
                    "plan_step_started",
                    "plan_step_completed",
                    "plan_completed",
                    "tool_started",
                    "tool_completed",
                    "tool_skipped",
                    "tool_failed",
                    "verification_started",
                    "verification_completed",
                    "approval_required",
                    "artifact_created",
                    "memory_operation_completed",
                    "heartbeat",
                }:
                    status.write(label)
                elif event_type in {"run_completed", "run_failed"}:
                    terminal_payload = payload
                    answer = str(payload.get("answer") or answer or payload.get("error") or "")
                    completed = event_type == "run_completed" and payload.get("status") == "completed"
                    status.update(
                        label="服务已完成" if completed else "服务处理未完成",
                        state="complete" if completed else "error",
                        expanded=False,
                    )
        except AgentApiError as exc:
            answer = f"服务暂时不可用：{exc}"
            status.update(label="服务连接异常", state="error", expanded=False)

        answer_placeholder.markdown(answer or "本次服务没有返回有效答案。")
        _render_audit_trail(event_log, request_id)

    approval_id = terminal_payload.get("approval_id")
    if approval_id:
        st.session_state.approval_id = approval_id
        st.session_state.pending_approval_request = {
            "approval_id": approval_id,
            "prompt": prompt,
            "session_id": st.session_state.session_id,
        }
        st.warning("该请求涉及受保护数据，正在等待授权审批。")
    elif terminal_payload.get("status") == "completed":
        st.session_state.pending_approval_request = {}
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer or "本次服务没有返回有效答案。",
            "request_id": request_id,
            "events": event_log,
        }
    )


def _render_memory(client: AgentApiClient) -> None:
    _page_heading(
        "Customer Memory",
        "客户记忆中心",
        "统一查看历史会话、短期上下文、情景记忆、长期事实、压缩摘要和已认证程序记忆。",
    )
    datasets = {}
    loaders = {
        "sessions": client.list_sessions,
        "events": client.memory_events,
        "summaries": client.memory_summaries,
        "procedures": client.memory_procedures,
    }
    for name, loader in loaders.items():
        try:
            datasets[name] = loader()
        except AgentApiError as exc:
            datasets[name] = []
            st.warning(f"{name} 暂时无法读取：{exc}")

    try:
        active_memories = client.list_memories(include_inactive=False)
    except AgentApiError as exc:
        active_memories = []
        st.warning(f"长期事实暂时无法读取：{exc}")

    tabs = st.tabs(["记忆总览", "历史会话", "情景记忆", "长期事实", "压缩摘要", "程序记忆"])

    with tabs[0]:
        metrics = st.columns(6)
        metrics[0].metric("历史会话", len(datasets["sessions"]))
        metrics[1].metric("短期消息", sum(int(item.get("message_count", 0)) for item in datasets["sessions"]))
        metrics[2].metric("情景记忆", len(datasets["events"]))
        metrics[3].metric("长期事实", len(active_memories))
        metrics[4].metric("压缩摘要", len(datasets["summaries"]))
        metrics[5].metric("程序记忆", len(datasets["procedures"]))
        st.info(
            "短期消息按会话连续使用；情景记忆用于跨会话召回；长期事实采用软衰减评分；"
            "程序记忆只有经过认证后才会展示和参与系统行为。"
        )

    with tabs[1]:
        sessions = datasets["sessions"]
        if not sessions:
            st.info("当前客户暂无可恢复的历史会话。完成一次问答后会自动出现在这里。")
        else:
            st.dataframe(
                [
                    {
                        "会话编号": item.get("session_id"),
                        "首个问题": item.get("title"),
                        "消息数": item.get("message_count"),
                        "开始时间": item.get("created_at"),
                        "最近更新": item.get("updated_at"),
                    }
                    for item in sessions
                ],
                use_container_width=True,
                hide_index=True,
            )
            session_ids = [str(item["session_id"]) for item in sessions]
            selected_session = st.selectbox(
                "选择要恢复的会话",
                session_ids,
                format_func=lambda value: next(
                    (
                        f"{item.get('title') or '未命名会话'} · {value[:8]}"
                        for item in sessions if item["session_id"] == value
                    ),
                    value,
                ),
            )
            if st.button("恢复并继续对话", type="primary", use_container_width=True):
                try:
                    payload = client.session_messages(selected_session)
                    st.session_state.session_id = selected_session
                    st.session_state.chat_messages = payload.get("messages", [])
                    st.session_state.last_request_id = ""
                    st.session_state.pending_navigation = "对话"
                    st.rerun()
                except AgentApiError as exc:
                    st.error(f"恢复会话失败：{exc}")

    with tabs[2]:
        events = datasets["events"]
        if not events:
            st.info("暂无情景记忆。成功完成的普通问答会自动形成情景记录。")
        else:
            st.dataframe(
                [
                    {
                        "时间": item.get("created_at"),
                        "用户问题": item.get("content"),
                        "助手结果": (item.get("metadata") or {}).get("assistant_message", ""),
                        "记忆判定": (
                            (item.get("metadata") or {}).get("memory_operation") or {}
                        ).get("status", ""),
                        "写入槽位": (
                            (item.get("metadata") or {}).get("memory_operation") or {}
                        ).get("saved_keys", []),
                        "待确认或失败原因": (
                            (item.get("metadata") or {}).get("memory_operation") or {}
                        ).get("failures", []),
                        "会话编号": item.get("session_id"),
                        "请求编号": item.get("request_id"),
                    }
                    for item in events
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        include_inactive = st.toggle(
            "显示非生效版本（待确认 / 已更正）", value=False
        )
        memories = active_memories
        if include_inactive:
            try:
                memories = client.list_memories(include_inactive=True)
            except AgentApiError as exc:
                st.error(f"读取历史版本失败：{exc}")
        list_column, edit_column = st.columns([2.15, 1], gap="large")
        with list_column:
            if memories:
                columns = [
                    "key", "value", "category", "status", "version", "confidence",
                    "importance", "scope", "source", "review_required",
                    "recency_score", "explicit", "last_confirmed_at",
                ]
                memory_rows = []
                for memory in memories:
                    metadata = memory.get("metadata") or {}
                    row = {key: memory.get(key) for key in columns}
                    row["scope"] = metadata.get("scope", "")
                    row["source"] = metadata.get("source", "")
                    row["review_required"] = metadata.get("review_required", False)
                    memory_rows.append(row)
                st.dataframe(
                    memory_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("暂无长期事实。可使用右侧表单，或在对话中说“请记住：我的设备型号是 S10”。")
        with edit_column:
            with st.form("memory_form", clear_on_submit=True):
                st.markdown("#### 显式新增或更正")
                key = st.text_input("记忆键", placeholder="例如 device.model")
                value = st.text_area("记忆内容", placeholder="例如 云鲸 S10")
                category_label = st.selectbox("记忆类别", list(MEMORY_CATEGORIES))
                importance = st.slider("重要度", 0.0, 1.0, 0.5, 0.05)
                submitted = st.form_submit_button("保存长期事实", use_container_width=True)
            if submitted:
                try:
                    saved = client.remember(key, value, MEMORY_CATEGORIES[category_label], importance)
                    st.success(f"已保存 {saved['key']}，版本 {saved['version']}")
                    st.rerun()
                except AgentApiError as exc:
                    st.error(f"保存失败：{exc}")
            active_keys = sorted({item["key"] for item in active_memories})
            selected_key = st.selectbox("遗忘指定事实", active_keys, disabled=not active_keys)
            if st.button("遗忘选中事实", disabled=not active_keys, use_container_width=True):
                try:
                    result = client.forget(selected_key)
                    st.success(f"已物理删除 {result.get('deleted', 0)} 条相关记忆")
                    st.rerun()
                except AgentApiError as exc:
                    st.error(f"删除失败：{exc}")
            pending_memories = [
                item
                for item in memories
                if item.get("status") == "pending_confirmation"
            ]
            if pending_memories:
                st.markdown("#### 待确认的冲突")
                pending_id = st.selectbox(
                    "选择候选记忆",
                    [item["memory_id"] for item in pending_memories],
                    format_func=lambda value: next(
                        (
                            f"{item.get('key')} → {item.get('value')}"
                            for item in pending_memories
                            if item["memory_id"] == value
                        ),
                        value,
                    ),
                )
                accept_column, reject_column = st.columns(2)
                if accept_column.button("接受新值", use_container_width=True):
                    try:
                        client.review_memory(pending_id, "accept")
                        st.success("已接受候选并生成新的生效版本。")
                        st.rerun()
                    except AgentApiError as exc:
                        st.error(f"确认失败：{exc}")
                if reject_column.button("保留原值", use_container_width=True):
                    try:
                        client.review_memory(pending_id, "reject")
                        st.success("已拒绝候选，原有记忆保持生效。")
                        st.rerun()
                    except AgentApiError as exc:
                        st.error(f"确认失败：{exc}")
            confirm_all = st.checkbox("确认遗忘该客户的全部记忆和关联情景")
            if st.button(
                "遗忘全部记忆", type="primary", disabled=not confirm_all,
                use_container_width=True,
            ):
                try:
                    result = client.forget()
                    st.success(f"已物理删除 {result.get('deleted', 0)} 条关联记录")
                    st.rerun()
                except AgentApiError as exc:
                    st.error(f"删除失败：{exc}")

    with tabs[4]:
        summaries = datasets["summaries"]
        if not summaries:
            st.info("暂无压缩摘要。同一会话达到摘要阈值后自动生成，原始短期消息与摘要分开保留。")
        else:
            st.dataframe(summaries, use_container_width=True, hide_index=True)

    with tabs[5]:
        procedures = datasets["procedures"]
        if not procedures:
            st.info("暂无已认证程序记忆。候选流程必须经 operator/admin 审批后才能在这里展示。")
        else:
            st.dataframe(procedures, use_container_width=True, hide_index=True)


def _render_operations(client: AgentApiClient) -> None:
    _page_heading(
        "Governance",
        "审批与服务产物",
        "处理受保护数据访问审批，并查询服务过程中生成的报告、证据和结构化产物。",
    )
    approval_tab, artifact_tab = st.tabs(["访问审批", "服务产物"])

    with approval_tab:
        if st.session_state.user_role in {"operator", "admin"}:
            st.caption("这里只展示当前租户的待审批数据访问请求。审批身份来自登录凭证。")
            try:
                pending_approvals = client.list_approvals(status="pending")
            except AgentApiError as exc:
                pending_approvals = []
                st.error(f"待办加载失败：{exc}")
            if not pending_approvals:
                st.info("当前没有待审批请求。")
            for approval in pending_approvals:
                approval_id = str(approval.get("approval_id") or "")
                args = approval.get("args") or {}
                with st.container(border=True):
                    st.markdown(f"#### 数据访问申请 · `{approval_id[:8]}`")
                    st.caption(
                        f"申请人 {approval.get('principal_id') or '历史记录'}"
                        f" · 工具 {approval.get('tool_name')}"
                        f" · 创建于 {approval.get('created_at')}"
                    )
                    st.write(
                        f"申请范围：用户 {args.get('user_id', '未知')}，"
                        f"月份 {args.get('month', '未知')}"
                    )
                    approve_column, deny_column = st.columns(2)
                    if approve_column.button(
                        "批准",
                        key=f"approve_{approval_id}",
                        use_container_width=True,
                    ):
                        try:
                            client.decide_approval(
                                approval_id,
                                "approve",
                                st.session_state.user_id,
                            )
                            st.success("已批准。申请人可以继续原请求。")
                            st.rerun()
                        except AgentApiError as exc:
                            st.error(f"批准失败：{exc}")
                    if deny_column.button(
                        "拒绝",
                        key=f"deny_{approval_id}",
                        use_container_width=True,
                    ):
                        try:
                            client.decide_approval(
                                approval_id,
                                "deny",
                                st.session_state.user_id,
                            )
                            st.success("已拒绝。")
                            st.rerun()
                        except AgentApiError as exc:
                            st.error(f"拒绝失败：{exc}")
        else:
            approval_id = str(st.session_state.approval_id or "")
            if not approval_id:
                st.info("当前账号没有待处理的数据访问申请。本人单月使用报告无需审批。")
            else:
                try:
                    approval = client.approval(approval_id)
                except AgentApiError as exc:
                    approval = None
                    st.error(f"审批状态查询失败：{exc}")
                if approval:
                    status = str(approval.get("status") or "pending")
                    status_labels = {
                        "pending": "等待审批",
                        "approved": "已批准",
                        "denied": "已拒绝",
                    }
                    with st.container(border=True):
                        st.markdown(f"#### {status_labels.get(status, status)}")
                        st.caption(f"审批编号 · {approval_id}")
                        st.json(approval.get("args") or {}, expanded=False)
                        pending_request = st.session_state.pending_approval_request
                        can_resume = (
                            status == "approved"
                            and pending_request.get("approval_id") == approval_id
                            and bool(pending_request.get("prompt"))
                        )
                        if st.button(
                            "继续原请求",
                            disabled=not can_resume,
                            use_container_width=True,
                        ):
                            st.session_state.session_id = pending_request["session_id"]
                            st.session_state.queued_prompt = pending_request["prompt"]
                            st.session_state.queued_approval_id = approval_id
                            st.session_state.navigation = "对话"
                            st.rerun()
                        if status == "pending":
                            st.info("请由 operator/admin 使用其独立凭证进入本页审批。")
                        elif status == "denied":
                            st.warning("该数据访问请求已被拒绝，不能继续执行。")

    with artifact_tab:
        with st.container(border=True):
            request_id = st.text_input(
                "服务请求编号",
                value=st.session_state.last_request_id,
                key="artifact_request_id",
            )
            if st.button("查询请求产物", disabled=not request_id, use_container_width=True):
                try:
                    st.json(client.artifacts(request_id))
                except AgentApiError as exc:
                    st.error(f"查询失败：{exc}")
            artifact_id = st.text_input("单个产物编号")
            if st.button("查询单个产物", disabled=not artifact_id, use_container_width=True):
                try:
                    st.json(client.artifact(artifact_id))
                except AgentApiError as exc:
                    st.error(f"查询失败：{exc}")


def _render_observability(client: AgentApiClient) -> None:
    _page_heading(
        "Service Reliability",
        "系统监控",
        "面向运维人员查看服务链路、运行指标、工具清单和健康状态。",
    )
    health_column, metrics_column, tools_column = st.columns(3)
    if health_column.button("服务健康检查", use_container_width=True):
        try:
            st.json(client.health())
        except AgentApiError as exc:
            st.error(str(exc))
    if metrics_column.button("查看指标快照", use_container_width=True):
        try:
            st.json(client.metrics_snapshot())
        except AgentApiError as exc:
            st.error(str(exc))
    if tools_column.button("查看工具清单", use_container_width=True):
        try:
            st.json(client.tool_manifest())
        except AgentApiError as exc:
            st.error(str(exc))

    with st.container(border=True):
        st.markdown("#### 服务链路查询")
        request_id = st.text_input(
            "请求编号",
            value=st.session_state.last_request_id,
            key="trace_request_id",
        )
        otel = st.checkbox("使用 OpenTelemetry 格式")
        if st.button("查询完整链路", disabled=not request_id):
            try:
                st.json(client.trace(request_id, otel=otel))
            except AgentApiError as exc:
                st.error(f"查询失败：{exc}")


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="智能客服",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    _init_state()
    selected = _sidebar_client()
    if selected is None:
        st.stop()
    page, client = selected

    renderers = {
        "chat": _render_chat,
        "memory": _render_memory,
        "operations": _render_operations,
        "observability": _render_observability,
    }
    renderers[page](client)


if __name__ == "__main__":
    main()
