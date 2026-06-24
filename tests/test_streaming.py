from utils.streaming import get_final_response


def test_final_response_returns_last_non_empty_chunk():
    assert get_final_response(["", "第一段\n", "最终回答\n"]) == "最终回答\n"


def test_final_response_uses_fallback_when_stream_is_empty():
    assert get_final_response([]) == "抱歉，本次没有生成有效回复，请稍后重试。"
