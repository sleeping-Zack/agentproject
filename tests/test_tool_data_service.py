from pathlib import Path

from services.tool_data_service import ToolDataService


def write_records(path: Path) -> None:
    path.write_text(
        '"用户ID","特征","清洁效率","耗材","对比","时间"\n'
        '"1001","65㎡公寓 | 单身 | 木地板","覆盖率:85%\\n日均清扫:45㎡",'
        '"主刷寿命:剩余60天","优于65%同面积用户","2025-06"\n',
        encoding="utf-8",
    )


def write_product_specs(path: Path) -> None:
    path.write_text(
        "model,product_name,category,suction_pa,battery_mah,runtime_minutes,"
        "navigation,dustbin_ml,water_tank_ml,features,catalog_version\n"
        "S10,DemoBot S10,扫地机器人,3000,3200,110,LDS,450,0,"
        "分区清扫|禁区设置,demo-v1\n",
        encoding="utf-8",
    )


def write_error_codes(path: Path) -> None:
    path.write_text(
        "model,error_code,title,meaning,actions,service_condition,safety_note,"
        "catalog_version\n"
        "S10,E12,驱动轮受阻,左轮被异物卡住,关机|清理轮组,仍报错时联系售后,"
        "不要自行拆机,demo-v1\n"
        "S20,E12,出水异常,出水通道受阻,检查水箱|清洁出水口,仍不出水时联系售后,"
        "不要用尖锐物疏通,demo-v1\n",
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


def test_product_specs_are_parsed_as_structured_values(tmp_path):
    records_path = tmp_path / "records.csv"
    specs_path = tmp_path / "product_specs.csv"
    error_codes_path = tmp_path / "error_codes.csv"
    write_records(records_path)
    write_product_specs(specs_path)
    write_error_codes(error_codes_path)
    service = ToolDataService(
        config={},
        records_path=str(records_path),
        product_specs_path=str(specs_path),
        error_codes_path=str(error_codes_path),
    )

    specs = service.get_product_specs(" s10 ")

    assert specs["model"] == "S10"
    assert specs["suction_pa"] == 3000
    assert specs["runtime_minutes"] == 110
    assert specs["features"] == ["分区清扫", "禁区设置"]
    assert specs["catalog_version"] == "demo-v1"


def test_error_codes_are_scoped_to_the_exact_model(tmp_path):
    records_path = tmp_path / "records.csv"
    specs_path = tmp_path / "product_specs.csv"
    error_codes_path = tmp_path / "error_codes.csv"
    write_records(records_path)
    write_product_specs(specs_path)
    write_error_codes(error_codes_path)
    service = ToolDataService(
        config={},
        records_path=str(records_path),
        product_specs_path=str(specs_path),
        error_codes_path=str(error_codes_path),
    )

    s10 = service.lookup_error_code("s10", " e12 ")
    s20 = service.lookup_error_code("S20", "E12")

    assert s10["meaning"] == "左轮被异物卡住"
    assert s20["meaning"] == "出水通道受阻"
    assert service.lookup_error_code("S10", "E99") == ""
    assert service.lookup_error_code("missing", "E12") == ""


def test_missing_product_specs_return_empty_string(tmp_path):
    records_path = tmp_path / "records.csv"
    specs_path = tmp_path / "product_specs.csv"
    error_codes_path = tmp_path / "error_codes.csv"
    write_records(records_path)
    write_product_specs(specs_path)
    write_error_codes(error_codes_path)
    service = ToolDataService(
        config={},
        records_path=str(records_path),
        product_specs_path=str(specs_path),
        error_codes_path=str(error_codes_path),
    )

    assert service.get_product_specs("missing") == ""


def test_real_product_specs_may_leave_unpublished_numbers_blank(tmp_path):
    records_path = tmp_path / "records.csv"
    specs_path = tmp_path / "product_specs.csv"
    write_records(records_path)
    specs_path.write_text(
        "model,product_name,category,suction_pa,battery_mah,runtime_minutes,"
        "navigation,dustbin_ml,water_tank_ml,features,catalog_version\n"
        "ROOMBA-105,Roomba 105,扫地机器人,,3000,,激光导航,,,"
        "障碍识别|自动回充,official-v1\n",
        encoding="utf-8",
    )
    service = ToolDataService(
        config={},
        records_path=str(records_path),
        product_specs_path=str(specs_path),
    )

    specs = service.get_product_specs("roomba-105")

    assert specs["suction_pa"] is None
    assert specs["battery_mah"] == 3000
    assert specs["dustbin_ml"] is None


def test_repository_official_product_and_fault_rows_are_queryable():
    service = ToolDataService(
        config={},
        records_path="data/external/records.csv",
        product_specs_path="data/external/product_specs.csv",
        error_codes_path="data/external/error_codes.csv",
    )

    product = service.get_product_specs("s8 maxv ultra")
    fault = service.lookup_error_code("t7s plus", "13")

    assert product["suction_pa"] == 10000
    assert product["battery_mah"] == 5200
    assert product["source_url"].startswith("https://us.roborock.com/")
    assert fault["title"] == "充电接触异常"
    assert fault["brand"] == "Roborock"
    assert fault["catalog_version"] == "official-2026-08-28"
