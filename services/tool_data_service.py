import csv
from datetime import datetime
from typing import Any, Dict


class ToolDataService:
    """Deterministic data provider behind agent tools.

    The app can later replace this class with real service clients without
    changing the LangChain tool signatures.
    """

    def __init__(self, config: Dict[str, Any], records_path: str) -> None:
        self.config = config
        self.records_path = records_path
        self._external_data: Dict[str, Dict[str, Dict[str, str]]] = {}

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
