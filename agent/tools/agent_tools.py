import json

from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from services.tool_data_service import ToolDataService
from agent.tools.registry import build_default_tool_registry

rag = RagSummarizeService()
tool_data_service = ToolDataService(
    config=agent_conf,
    records_path=get_abs_path(agent_conf["external_data_path"]),
)
tool_registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))


def _require_allowed(tool_name: str) -> None:
    tool_registry.require_allowed(tool_name)


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    _require_allowed("rag_summarize")
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    _require_allowed("get_weather")
    return tool_data_service.get_weather(city)


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    _require_allowed("get_user_location")
    return tool_data_service.get_user_location()


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    _require_allowed("get_user_id")
    return tool_data_service.get_user_id()


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    _require_allowed("get_current_month")
    return tool_data_service.get_current_month()


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    _require_allowed("fetch_external_data")
    record = tool_data_service.fetch_external_data(user_id, month)
    if not record:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    _require_allowed("fill_context_for_report")
    return "fill_context_for_report已调用"
