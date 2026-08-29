import json
from dataclasses import asdict
from typing import Literal

from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from services.tool_data_service import ToolDataService
from services.factories import create_approval_store, create_artifact_store
from agent.tools.registry import build_default_tool_registry
from safety.security import (
    assert_safe_tool_arguments,
    is_sensitive_tool_approved,
    require_sensitive_tool_confirmation,
)
from observability.context import request_context
from observability.tracing import trace_recorder

rag = RagSummarizeService()
tool_data_service = ToolDataService(
    config=agent_conf,
    records_path=get_abs_path(agent_conf["external_data_path"]),
    product_specs_path=get_abs_path(agent_conf["product_specs_path"]),
    error_codes_path=get_abs_path(agent_conf["error_codes_path"]),
)
artifact_store = create_artifact_store()
approval_store = create_approval_store()
tool_registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))


def _require_allowed(tool_name: str) -> None:
    tool_registry.require_allowed(tool_name)


@tool(description="从向量存储中检索参考资料；每次检索必须说明尚未覆盖的信息缺口")
def rag_summarize(query: str, information_gap: str) -> str:
    _require_allowed("rag_summarize")
    assert_safe_tool_arguments(
        "rag_summarize",
        {"query": query, "information_gap": information_gap},
    )
    result = rag.rag_summarize_result(query)
    _record_rag_evidence(result)
    # verification_failed 时，rag_service 已保留带引用的摘要（citation_validity 达标），
    # 优先返回该摘要供下游正常引用；仅当摘要为空时才降级为裸原文。
    if result.business_status == "verification_failed" and result.evidence and not result.answer.strip():
        return _render_rag_evidence(result.evidence)
    return result.answer


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    _require_allowed("get_weather")
    assert_safe_tool_arguments("get_weather", {"city": city})
    return tool_data_service.get_weather(city)


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    _require_allowed("get_user_location")
    return tool_data_service.get_user_location()


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    _require_allowed("get_user_id")
    return request_context().extra.get("data_user_id") or tool_data_service.get_user_id()


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    _require_allowed("get_current_month")
    return tool_data_service.get_current_month()


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    _require_allowed("fetch_external_data")
    assert_safe_tool_arguments("fetch_external_data", {"user_id": user_id, "month": month})
    require_sensitive_tool_confirmation(
        "fetch_external_data",
        confirmed=is_sensitive_tool_approved("fetch_external_data"),
    )
    record = tool_data_service.fetch_external_data(user_id, month)
    if not record:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
    return json.dumps(record, ensure_ascii=False)


@tool(description="按设备型号和故障码精确查询诊断、安全步骤及转售后条件")
def lookup_error_code(model: str, error_code: str) -> str:
    _require_allowed("lookup_error_code")
    arguments = {"model": model, "error_code": error_code}
    assert_safe_tool_arguments("lookup_error_code", arguments)
    result = tool_data_service.lookup_error_code(model, error_code)
    return json.dumps(result, ensure_ascii=False) if result else ""


@tool(description="按设备型号精确查询结构化产品规格；目录未收录时返回空字符串")
def get_product_specs(model: str) -> str:
    _require_allowed("get_product_specs")
    arguments = {"model": model}
    assert_safe_tool_arguments("get_product_specs", arguments)
    result = tool_data_service.get_product_specs(model)
    return json.dumps(result, ensure_ascii=False) if result else ""


@tool(description="创建售后工单；仅在用户明确要求且审批通过后执行")
def create_support_ticket(
    model: str,
    issue_type: Literal["repair", "maintenance", "warranty", "other"],
    description: str,
    error_code: str = "",
) -> str:
    _require_allowed("create_support_ticket")
    arguments = _normalize_ticket_arguments(
        {
            "model": model,
            "issue_type": issue_type,
            "description": description,
            "error_code": error_code,
        }
    )
    if not arguments["model"] or not arguments["description"]:
        raise ValueError("设备型号和问题描述不能为空")
    if arguments["issue_type"] not in {"repair", "maintenance", "warranty", "other"}:
        raise ValueError("不支持的工单问题类型")
    assert_safe_tool_arguments("create_support_ticket", arguments)
    require_sensitive_tool_confirmation(
        "create_support_ticket",
        confirmed=is_sensitive_tool_approved("create_support_ticket"),
    )

    ctx = request_context()
    approval_id = str(ctx.extra.get("approval_id") or "").strip()
    if not approval_id:
        raise PermissionError("创建售后工单需要有效审批记录")
    tenant_id = str(ctx.tenant_id or "").strip()
    principal_id = str(ctx.extra.get("user_id") or "").strip()
    if not tenant_id or not principal_id:
        raise PermissionError("创建售后工单需要已认证的租户和用户上下文")
    try:
        approval = approval_store.get(approval_id)
    except KeyError as exc:
        raise PermissionError("售后工单审批记录不存在") from exc
    if (
        not approval.is_approved
        or approval.tenant_id != tenant_id
        or approval.principal_id != principal_id
        or approval.tool_name != "create_support_ticket"
        or _normalize_ticket_arguments(approval.args) != arguments
    ):
        raise PermissionError("售后工单审批记录与当前调用不匹配或尚未批准")
    payload = {
        "tenant_id": tenant_id,
        "principal_id": principal_id,
        **arguments,
        "status": "open",
    }
    artifact = artifact_store.save_artifact(
        request_id=approval_id,
        tenant_id=tenant_id,
        artifact_type="support_ticket",
        name="created-ticket",
        payload=payload,
        metadata={
            "approval_id": approval_id,
            "execution_request_id": ctx.request_id or "",
        },
    )
    if artifact.payload != payload:
        raise ValueError("同一审批记录已用于不同的售后工单内容")
    return json.dumps(
        {
            "ticket_id": artifact.artifact_id,
            "status": artifact.payload["status"],
            "model": artifact.payload["model"],
            "error_code": artifact.payload["error_code"],
            "created_at": artifact.created_at,
        },
        ensure_ascii=False,
    )


def _normalize_ticket_arguments(arguments: dict) -> dict:
    return {
        "model": str(arguments.get("model") or "").strip().upper(),
        "issue_type": str(arguments.get("issue_type") or "").strip().lower(),
        "description": str(arguments.get("description") or "").strip(),
        "error_code": str(arguments.get("error_code") or "").strip().upper(),
    }


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    _require_allowed("fill_context_for_report")
    return "fill_context_for_report已调用"


REACT_TOOLS = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
    lookup_error_code,
    get_product_specs,
    create_support_ticket,
]


def _record_rag_evidence(result) -> None:
    ctx = request_context()
    if not ctx.request_id:
        return
    evidence = [asdict(item) for item in result.evidence]
    try:
        with trace_recorder.span(
            ctx.request_id,
            category="rag",
            name="evidence",
            metadata={
                "business_status": result.business_status,
                "evidence": evidence,
                "verification": dict(result.verification or {}),
            },
        ):
            pass
    except KeyError:
        return


def _render_rag_evidence(evidence) -> str:
    lines = ["生成式总结未通过一致性校验，请仅依据以下知识库原文作答："]
    for item in list(evidence)[:5]:
        payload = asdict(item)
        evidence_id = str(payload.get("id") or "evidence")
        source = str(payload.get("source") or "知识库")
        content = str(payload.get("content") or "").strip()[:800]
        if content:
            lines.append(f"- [{evidence_id}] {source}：{content}")
    return "\n".join(lines)
