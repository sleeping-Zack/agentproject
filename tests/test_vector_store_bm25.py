from pathlib import Path

from rag.vector_store import VectorStoreService


def test_bm25_index_path_reads_nested_retrieval_config():
    service = VectorStoreService.__new__(VectorStoreService)

    path = service._bm25_index_path()

    assert path is not None
    assert Path(path).name == "bm25_index.pkl"
    assert Path(path).is_absolute()
