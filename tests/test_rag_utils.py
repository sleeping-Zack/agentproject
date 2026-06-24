from rag.rag_utils import build_document_metadata, format_citations, hybrid_rank


class Doc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def test_build_document_metadata_tracks_source_and_chunk_version(tmp_path):
    source = tmp_path / "维护保养.txt"
    source.write_text("主刷需要定期清理", encoding="utf-8")

    metadata = build_document_metadata(str(source), chunk_version="v2")

    assert metadata["source_name"] == "维护保养.txt"
    assert metadata["source_path"].endswith("维护保养.txt")
    assert metadata["chunk_version"] == "v2"
    assert len(metadata["content_hash"]) == 32


def test_hybrid_rank_uses_keyword_signal_with_vector_score():
    docs = [
        Doc("扫地机器人无法连接 WiFi 时应重启路由器", {"doc_id": "wifi"}),
        Doc("主刷缠绕毛发时应剪断并清理滚刷", {"doc_id": "brush"}),
    ]

    ranked = hybrid_rank(
        "主刷 缠绕",
        docs,
        vector_scores={"wifi": 0.9, "brush": 0.1},
        keyword_weight=0.8,
        top_n=1,
    )

    assert ranked[0].metadata["doc_id"] == "brush"


def test_format_citations_includes_source_metadata():
    docs = [
        Doc("主刷缠绕毛发时应清理滚刷", {"source_name": "故障排除.txt", "page": 1}),
    ]

    citations = format_citations(docs)

    assert citations == "[1] 故障排除.txt#page=1"
