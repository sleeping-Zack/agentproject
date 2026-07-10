"""BM25Retriever：build/save/load/retrieve 走通，中文分词生效。

依赖 jieba 和 rank_bm25，如果没装就直接 skip 整个文件。
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

pytest.importorskip("jieba")
pytest.importorskip("rank_bm25")

from rag.retrievers.bm25_retriever import BM25Retriever


def _docs():
    return [
        Document(page_content="主刷缠绕毛发时应剪断并清理滚刷", metadata={"doc_id": "brush"}),
        Document(page_content="扫地机器人无法连接 WiFi 时应重启路由器", metadata={"doc_id": "wifi"}),
        Document(page_content="尘盒滤网每周清理一次可以延长寿命", metadata={"doc_id": "filter"}),
    ]


def test_bm25_build_then_retrieve_returns_relevant_doc():
    retriever = BM25Retriever()
    retriever.build(_docs())
    assert retriever.is_ready()

    result = retriever.retrieve("主刷缠绕毛发怎么办", k=3)
    assert result, "BM25 未召回任何文档"
    assert result[0].doc_id == "brush"
    assert result[0].sparse_score > 0


def test_bm25_returns_empty_when_not_ready():
    retriever = BM25Retriever()
    assert retriever.retrieve("任何查询") == []


def test_bm25_save_and_load_roundtrip(tmp_path):
    index_path = tmp_path / "bm25.pkl"
    retriever = BM25Retriever(index_path=str(index_path))
    retriever.build(_docs())
    retriever.save()

    reloaded = BM25Retriever(index_path=str(index_path))
    assert reloaded.load()
    result = reloaded.retrieve("WiFi 连接", k=2)
    assert result[0].doc_id == "wifi"
