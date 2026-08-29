from observability.context import bind_request_context, request_context
from utils.prompt_loader import (load_prompt_document, load_rag_prompts,
                                  load_report_prompts, load_system_prompts)


def test_load_prompt_document_parses_frontmatter():
    doc = load_prompt_document("main")
    assert doc.name == "main"
    assert doc.version == "v6"
    assert any("fill_context_for_report" in entry for entry in doc.changelog)
    assert any("售后工单" in entry for entry in doc.changelog)


def test_load_prompt_document_returns_clean_body():
    doc = load_prompt_document("rag_summarize")
    assert not doc.content.startswith("---")
    assert "基于参考资料总结" in doc.content


def test_loading_prompt_records_version_in_context():
    with bind_request_context(request_id="r1"):
        load_system_prompts()
        ctx = request_context()
        assert ctx.prompt_version == "main:v6"


def test_main_prompt_guards_demo_catalog_and_ticket_creation():
    content = load_prompt_document("main").content

    assert "必须先追问缺失信息" in content
    assert "不得猜测、套用默认型号或借用其他型号的数据" in content
    assert "不代表任何真实品牌或厂商官方资料" in content
    assert "仅当用户明确要求创建、提交或开立售后工单" in content
    assert "只有工具返回 ticket_id 后才能声称创建成功" in content


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
