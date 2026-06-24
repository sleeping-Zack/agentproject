from observability.context import bind_request_context, request_context
from utils.prompt_loader import (load_prompt_document, load_rag_prompts,
                                  load_report_prompts, load_system_prompts)


def test_load_prompt_document_parses_frontmatter():
    doc = load_prompt_document("main")
    assert doc.name == "main"
    assert doc.version == "v2"
    assert any("fill_context_for_report" in entry for entry in doc.changelog)


def test_load_prompt_document_returns_clean_body():
    doc = load_prompt_document("rag_summarize")
    assert not doc.content.startswith("---")
    assert "基于参考资料总结" in doc.content


def test_loading_prompt_records_version_in_context():
    with bind_request_context(request_id="r1"):
        load_system_prompts()
        ctx = request_context()
        assert ctx.prompt_version == "main:v2"


def test_all_three_prompts_are_versioned():
    main_doc = load_prompt_document("main")
    rag_doc = load_prompt_document("rag_summarize")
    report_doc = load_prompt_document("report")
    assert main_doc.version != "unversioned"
    assert rag_doc.version != "unversioned"
    assert report_doc.version != "unversioned"


def test_loaders_return_string_body():
    assert isinstance(load_system_prompts(), str)
    assert isinstance(load_rag_prompts(), str)
    assert isinstance(load_report_prompts(), str)
