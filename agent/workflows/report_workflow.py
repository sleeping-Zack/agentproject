from typing import Any, Dict


class ReportWorkflow:
    """Explicit report-generation workflow.

    The class is intentionally deterministic for tests. In runtime it can be
    wrapped by LangGraph, but the step boundaries remain visible.
    """

    def __init__(self, tool_service, rag_service) -> None:
        self.tool_service = tool_service
        self.rag_service = rag_service

    def run(
        self,
        query: str,
        *,
        user_id: str | None = None,
        month: str | None = None,
        tenant_id: str | None = None,
        intent: str | None = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "query": query,
            "requested_user_id": user_id,
            "requested_month": month,
            "tenant_id": tenant_id,
            "requested_intent": intent,
        }
        self._detect_intent(state)
        if state["intent"] != "report":
            self._fallback(state, "当前工作流只处理个人使用报告。")
            return state
        self._load_user_context(state)
        self._fetch_record(state)
        if not state.get("record"):
            self._fallback(state, "没有找到对应月份的使用记录，暂时无法生成报告。")
            return state
        self._rag_supplement(state)
        self._generate_report(state)
        return state

    def _detect_intent(self, state: Dict[str, Any]) -> None:
        if state.get("requested_intent") == "report":
            state["intent"] = "report"
            return
        query = state["query"]
        state["intent"] = "report" if "报告" in query or "使用记录" in query else "qa"

    def _load_user_context(self, state: Dict[str, Any]) -> None:
        state["user_id"] = state.get("requested_user_id") or self.tool_service.get_user_id()
        state["month"] = state.get("requested_month") or self.tool_service.get_current_month()

    def _fetch_record(self, state: Dict[str, Any]) -> None:
        state["record"] = self.tool_service.fetch_external_data(state["user_id"], state["month"])

    def _rag_supplement(self, state: Dict[str, Any]) -> None:
        result_method = getattr(self.rag_service, "rag_summarize_result", None)
        if callable(result_method):
            result = result_method(
                "扫地机器人使用报告保养建议",
                tenant_id=state.get("tenant_id"),
            )
            verification = result.verification or {}
            if verification.get("passed", True) is False:
                state["rag_advice"] = "暂无经过证据一致性校验的补充建议。"
                state["rag_advice_status"] = "unavailable"
            else:
                state["rag_advice"] = result.answer
                state["rag_advice_status"] = "available"
            state["evidence"] = list(result.evidence)
            return
        state["rag_advice"] = self.rag_service.rag_summarize("扫地机器人使用报告保养建议")
        state["evidence"] = []

    def _generate_report(self, state: Dict[str, Any]) -> None:
        record = state["record"]
        state["fallback"] = False
        state["answer"] = (
            "# 黑马程序员扫地机器人使用情况报告与保养建议\n\n"
            f"- 用户：{state['user_id']}\n"
            f"- 月份：{state['month']}\n"
            f"- 使用特征：{record.get('特征', '')}\n"
            f"- 清洁效率：{record.get('效率', '')}\n"
            f"- 补充建议：{state['rag_advice']}"
        )

    def _fallback(self, state: Dict[str, Any], message: str) -> None:
        state["fallback"] = True
        state["answer"] = message
