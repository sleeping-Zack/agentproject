from pathlib import Path


def test_main_prompt_does_not_request_hidden_chain_of_thought():
    prompt = Path("prompts/main_prompt.txt").read_text(encoding="utf-8")

    assert "真实的自然语言思考过程" not in prompt
    assert "简要说明工具调用原因" in prompt


def test_main_prompt_preserves_rag_evidence_ids_in_final_answer():
    prompt = Path("prompts/main_prompt.txt").read_text(encoding="utf-8")

    assert "原样保留" in prompt
    assert "RAG 证据 ID" in prompt
    assert "禁止删除、重新编号或伪造" in prompt
