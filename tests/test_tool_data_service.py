from pathlib import Path

from services.tool_data_service import ToolDataService


def write_records(path: Path) -> None:
    path.write_text(
        '"用户ID","特征","清洁效率","耗材","对比","时间"\n'
        '"1001","65㎡公寓 | 单身 | 木地板","覆盖率:85%\\n日均清扫:45㎡",'
        '"主刷寿命:剩余60天","优于65%同面积用户","2025-06"\n',
        encoding="utf-8",
    )


def test_runtime_context_is_deterministic(tmp_path):
    records_path = tmp_path / "records.csv"
    write_records(records_path)
    service = ToolDataService(
        config={
            "default_user_id": "1001",
            "default_user_location": "深圳",
            "current_month": "2025-06",
            "weather": {
                "深圳": {
                    "condition": "晴",
                    "temperature_c": 29,
                    "humidity": 70,
                    "wind": "南风2级",
                    "aqi": 30,
                    "rain_probability": "低",
                }
            },
        },
        records_path=str(records_path),
    )

    assert service.get_user_id() == "1001"
    assert service.get_user_location() == "深圳"
    assert service.get_current_month() == "2025-06"
    assert "城市深圳天气为晴" in service.get_weather("深圳")


def test_external_records_use_csv_parser(tmp_path):
    records_path = tmp_path / "records.csv"
    write_records(records_path)
    service = ToolDataService(
        config={"default_user_id": "1001", "default_user_location": "深圳", "current_month": "2025-06"},
        records_path=str(records_path),
    )

    record = service.fetch_external_data("1001", "2025-06")

    assert record == {
        "特征": "65㎡公寓 | 单身 | 木地板",
        "效率": "覆盖率:85%\\n日均清扫:45㎡",
        "耗材": "主刷寿命:剩余60天",
        "对比": "优于65%同面积用户",
    }


def test_missing_external_record_returns_empty_string(tmp_path):
    records_path = tmp_path / "records.csv"
    write_records(records_path)
    service = ToolDataService(
        config={"default_user_id": "1001", "default_user_location": "深圳", "current_month": "2025-06"},
        records_path=str(records_path),
    )

    assert service.fetch_external_data("9999", "2025-06") == ""
