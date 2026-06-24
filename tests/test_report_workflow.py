from agent.workflows.report_workflow import ReportWorkflow


class FakeToolService:
    def get_user_id(self):
        return "1005"

    def get_current_month(self):
        return "2025-09"

    def fetch_external_data(self, user_id, month):
        return {"特征": "120㎡ | 老人 | 防滑砖", "效率": "定时清扫使用率:4次/周"}


class FakeRagService:
    def rag_summarize(self, query):
        return "建议关注电池保养。"


def test_report_workflow_runs_explicit_steps():
    workflow = ReportWorkflow(tool_service=FakeToolService(), rag_service=FakeRagService())

    result = workflow.run("帮我生成本月使用报告")

    assert result["intent"] == "report"
    assert result["user_id"] == "1005"
    assert result["month"] == "2025-09"
    assert "定时清扫使用率" in result["record"]["效率"]
    assert "建议关注电池保养" in result["answer"]


def test_report_workflow_fallback_for_missing_record():
    class MissingRecordService(FakeToolService):
        def fetch_external_data(self, user_id, month):
            return ""

    workflow = ReportWorkflow(tool_service=MissingRecordService(), rag_service=FakeRagService())

    result = workflow.run("帮我生成本月使用报告")

    assert result["fallback"] is True
    assert "没有找到" in result["answer"]
