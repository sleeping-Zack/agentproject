import re


class UnsafeInputError(ValueError):
    pass


INJECTION_PATTERNS = [
    re.compile(r"忽略.*(指令|规则|系统提示词)", re.IGNORECASE),
    re.compile(r"泄露.*(系统提示词|system prompt|prompt)", re.IGNORECASE),
    re.compile(r"ignore .*previous .*instructions", re.IGNORECASE),
    re.compile(r"reveal .*system prompt", re.IGNORECASE),
]

RAG_INJECTION_PATTERNS = [
    re.compile(r"忽略.*(系统提示词|开发者指令|工具规则)", re.IGNORECASE),
    re.compile(r"调用\s*(fetch_external_data|get_user_id)", re.IGNORECASE),
    re.compile(r"ignore .*system .*prompt", re.IGNORECASE),
]

TOOL_ARGUMENT_RULES = {
    "get_weather": {"city": re.compile(r"^[\u4e00-\u9fa5A-Za-z\s-]{1,30}$")},
    "rag_summarize": {"query": re.compile(r"^.{1,200}$", re.DOTALL)},
    "fetch_external_data": {
        "user_id": re.compile(r"^\d{4,20}$"),
        "month": re.compile(r"^\d{4}-\d{2}$"),
    },
}

SENSITIVE_TOOLS = {"fetch_external_data"}

SECRET_PATTERNS = [
    re.compile(r"(DASHSCOPE_API_KEY=)[^\s]+", re.IGNORECASE),
    re.compile(r"(OPENAI_API_KEY=)[^\s]+", re.IGNORECASE),
    re.compile(r"(token\s*[:=]\s*)[^\s,}]+", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)[^\s,}]+", re.IGNORECASE),
]


def is_prompt_injection(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in INJECTION_PATTERNS)


def assert_safe_user_input(text: str) -> None:
    if is_prompt_injection(text):
        raise UnsafeInputError("输入疑似包含越权或提示词注入请求")


def assert_safe_retrieved_content(text: str) -> None:
    if any(pattern.search(text or "") for pattern in RAG_INJECTION_PATTERNS):
        raise UnsafeInputError("检索内容疑似包含提示词注入或越权工具调用指令")


def assert_safe_tool_arguments(tool_name: str, arguments: dict) -> None:
    rules = TOOL_ARGUMENT_RULES.get(tool_name, {})
    for key, pattern in rules.items():
        value = str(arguments.get(key, ""))
        if not pattern.match(value):
            raise UnsafeInputError(f"工具参数非法：{tool_name}.{key}")


def require_sensitive_tool_confirmation(tool_name: str, confirmed: bool) -> None:
    if tool_name in SENSITIVE_TOOLS and not confirmed:
        raise PermissionError(f"敏感工具需要确认后才能调用：{tool_name}")


def redact_sensitive(value):
    if isinstance(value, dict):
        return {key: redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if not isinstance(value, str):
        return value

    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}<redacted>", redacted)
    if "secret" in redacted.lower() and redacted != "<redacted>":
        return "<redacted>"
    return redacted
