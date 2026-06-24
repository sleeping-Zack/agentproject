from pathlib import Path


def test_main_prompt_does_not_request_hidden_chain_of_thought():
    prompt = Path("prompts/main_prompt.txt").read_text(encoding="utf-8")

    assert "真实的自然语言思考过程" not in prompt
    assert "简要说明工具调用原因" in prompt
