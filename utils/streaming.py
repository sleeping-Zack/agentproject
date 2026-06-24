DEFAULT_EMPTY_RESPONSE = "抱歉，本次没有生成有效回复，请稍后重试。"


def get_final_response(chunks, fallback: str = DEFAULT_EMPTY_RESPONSE) -> str:
    for chunk in reversed(chunks):
        if chunk:
            return chunk
    return fallback
