from langchain_core.documents import Document

from rag.schemas import RetrievalCandidate, stable_doc_id


def test_final_score_preserves_a_real_zero_rerank_score():
    candidate = RetrievalCandidate(
        doc_id="doc",
        document=Document(page_content="content"),
        dense_score=0.8,
        fusion_score=0.2,
        rerank_score=0.0,
    )

    assert candidate.final_score() == 0.0


def test_stable_doc_id_does_not_depend_on_fallback_rank():
    document = Document(page_content="同一个没有 metadata 的旧文档")

    assert stable_doc_id(document, 1) == stable_doc_id(document, 99)


def test_stable_doc_id_normalizes_absolute_source_path():
    document = Document(
        page_content="content",
        metadata={"source": "C:/deploy/data/manual.txt", "chunk_index": 3},
    )

    assert stable_doc_id(document) == "manual.txt#3"
