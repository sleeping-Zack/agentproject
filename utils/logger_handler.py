import json
import logging
import os
from datetime import datetime

from observability.context import request_context
from safety.security import redact_sensitive
from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

_RESERVED_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """一行一事件 JSON 日志，自动注入当前 ContextVar 中的请求级字段。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "file": f"{record.filename}:{record.lineno}",
            "msg": redact_sensitive(record.getMessage()),
        }
        ctx = request_context().as_dict()
        if ctx:
            payload.update(ctx)
        for key, value in record.__dict__.items():
            if key in _RESERVED_FIELDS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive(str(record.msg))
        return True


def _select_formatter() -> logging.Formatter:
    fmt = os.getenv("AGENT_LOG_FORMAT", "json").lower()
    if fmt == "text":
        return DEFAULT_LOG_FORMAT
    return JsonFormatter()


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = _select_formatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    if isinstance(formatter, JsonFormatter):
        # JSON 已脱敏 message，避免双重处理
        pass
    else:
        console_handler.addFilter(RedactionFilter())
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    if not isinstance(formatter, JsonFormatter):
        file_handler.addFilter(RedactionFilter())
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
