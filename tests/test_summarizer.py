from agent.summarizer import ConversationSummarizer


def test_summarizer_invokes_callable_and_returns_text():
    received = {}

    def invoker(prompt: str) -> str:
        received["prompt"] = prompt
        return "压缩后的摘要"

    summarizer = ConversationSummarizer(invoker=invoker)
    history = [
        {"role": "user", "content": "我家是小户型"},
        {"role": "assistant", "content": "推荐小型扫地机器人"},
    ]

    summary = summarizer(history, previous_summary="之前提到小户型")

    assert summary == "压缩后的摘要"
    assert "小户型" in received["prompt"]
    assert "之前提到小户型" in received["prompt"]


def test_summarizer_returns_previous_on_empty_history():
    summarizer = ConversationSummarizer(invoker=lambda _: "won't be called")

    assert summarizer([], previous_summary="保留") == "保留"


def test_summarizer_swallows_invoker_errors():
    def broken(_prompt):
        raise RuntimeError("network")

    summarizer = ConversationSummarizer(invoker=broken)
    history = [{"role": "user", "content": "test"}]

    assert summarizer(history, previous_summary="fallback") == "fallback"
