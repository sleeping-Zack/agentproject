
from services.tool_data_service import ToolDataService


def test_report_flow_fetches_configured_user_current_month(tmp_path):
    records_path = tmp_path / "records.csv"
    records_path.write_text(
        '"用户ID","特征","清洁效率","耗材","对比","时间"\n'
        '"1005","120㎡ | 老人 | 防滑砖","定时清扫使用率:4次/周","电池衰减:28%","操作教学中","2025-09"\n',
        encoding="utf-8",
    )
    service = ToolDataService(
        config={
            "default_user_id": "1005",
            "default_user_location": "合肥",
            "current_month": "2025-09",
        },
        records_path=str(records_path),
    )

    user_id = service.get_user_id()
    month = service.get_current_month()
    record = service.fetch_external_data(user_id, month)

    assert record["特征"] == "120㎡ | 老人 | 防滑砖"
    assert record["效率"] == "定时清扫使用率:4次/周"
