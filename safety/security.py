import re


class UnsafeInputError(ValueError):
    pass


INJECTION_PATTERNS = [
    re.compile(r"忽略.*(指令|规则|系统提示词)", re.IGNORECASE),
    re.compile(r"泄露.*(系统提示词|system prompt|prompt)", re.IGNORECASE),
    re.compile(r"ignore .*previous .*instructions", re.IGNORECASE),
    re.compile(r"reveal .*system prompt", re.IGNORECASE),
]

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
