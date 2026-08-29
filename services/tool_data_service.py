import csv
from datetime import datetime
from typing import Any, Dict, Optional


class ToolDataService:
    """Deterministic data provider behind agent tools.

    The app can later replace this class with real service clients without
    changing the LangChain tool signatures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        records_path: str,
        product_specs_path: Optional[str] = None,
        error_codes_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.records_path = records_path
        self.product_specs_path = product_specs_path
        self.error_codes_path = error_codes_path
        self._external_data: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._product_specs: Dict[str, Dict[str, Any]] = {}
        self._error_codes: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._product_specs_loaded = False
        self._error_codes_loaded = False

    def get_user_id(self) -> str:
        return str(self.config.get("default_user_id", "1001"))

    def get_user_location(self) -> str:
        return str(self.config.get("default_user_location", "深圳"))

    def get_current_month(self) -> str:
        configured_month = self.config.get("current_month")
        if configured_month:
            return str(configured_month)
        return datetime.now().strftime("%Y-%m")

    def get_weather(self, city: str) -> str:
        weather_by_city = self.config.get("weather", {})
        weather = weather_by_city.get(city) or self.config.get("default_weather", {})
        condition = weather.get("condition", "晴天")
        temperature = weather.get("temperature_c", 26)
        humidity = weather.get("humidity", 50)
        wind = weather.get("wind", "南风1级")
        aqi = weather.get("aqi", 21)
        rain_probability = weather.get("rain_probability", "极低")
        return (
            f"城市{city}天气为{condition}，气温{temperature}摄氏度，"
            f"空气湿度{humidity}%，{wind}，AQI{aqi}，最近6小时降雨概率{rain_probability}"
        )

    def fetch_external_data(self, user_id: str, month: str):
        self._load_external_data()
        return self._external_data.get(user_id, {}).get(month, "")

    def get_product_specs(self, model: str):
        self._load_product_specs()
        return self._product_specs.get(self._normalize_identifier(model), "")

    def lookup_error_code(self, model: str, error_code: str):
        self._load_error_codes()
        key = (
            self._normalize_identifier(model),
            self._normalize_identifier(error_code),
        )
        return self._error_codes.get(key, "")

    def _load_external_data(self) -> None:
        if self._external_data:
            return

        with open(self.records_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row["用户ID"]
                month = row["时间"]
                self._external_data.setdefault(user_id, {})[month] = {
                    "特征": row["特征"],
                    "效率": row["清洁效率"],
                    "耗材": row["耗材"],
                    "对比": row["对比"],
                }

    def _load_product_specs(self) -> None:
        if self._product_specs_loaded:
            return
        self._product_specs_loaded = True
        if not self.product_specs_path:
            return

        integer_fields = (
            "suction_pa",
            "battery_mah",
            "runtime_minutes",
            "dustbin_ml",
            "water_tank_ml",
        )
        with open(self.product_specs_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item: Dict[str, Any] = dict(row)
                for field in integer_fields:
                    value = str(item.get(field) or "").strip()
                    item[field] = int(value) if value else None
                item["features"] = self._split_list(item.get("features", ""))
                model = self._normalize_identifier(item["model"])
                item["model"] = model
                self._product_specs[model] = item

    def _load_error_codes(self) -> None:
        if self._error_codes_loaded:
            return
        self._error_codes_loaded = True
        if not self.error_codes_path:
            return

        with open(self.error_codes_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item: Dict[str, Any] = dict(row)
                model = self._normalize_identifier(item["model"])
                error_code = self._normalize_identifier(item["error_code"])
                item["model"] = model
                item["error_code"] = error_code
                item["actions"] = self._split_list(item.get("actions", ""))
                self._error_codes[(model, error_code)] = item

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _split_list(value: str) -> list[str]:
        return [item.strip() for item in str(value or "").split("|") if item.strip()]
