from typing import Any, Dict


class ReportWorkflow:
    """Explicit report-generation workflow.

    The class is intentionally deterministic for tests. In runtime it can be
    wrapped by LangGraph, but the step boundaries remain visible.
    """

    def __init__(self, tool_service, rag_service) -> None:
        self.tool_service = tool_service
        self.rag_service = rag_service

    def run(self, query: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {"query": query}
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
        query = state["query"]
        state["intent"] = "report" if "报告" in query or "使用记录" in query else "qa"

    def _load_user_context(self, state: Dict[str, Any]) -> None:
        state["user_id"] = self.tool_service.get_user_id()
        state["month"] = self.tool_service.get_current_month()

    def _fetch_record(self, state: Dict[str, Any]) -> None:
        state["record"] = self.tool_service.fetch_external_data(state["user_id"], state["month"])

    def _rag_supplement(self, state: Dict[str, Any]) -> None:
        state["rag_advice"] = self.rag_service.rag_summarize("扫地机器人使用报告保养建议")

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
