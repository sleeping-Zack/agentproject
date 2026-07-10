"""
总结服务类：用户提问，走 Hybrid 检索（Dense + BM25 + RRF + 可选 Rerank），把证据交给模型总结。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model, embed_model
from observability.metrics import metrics_registry
from rag.rag_utils import format_citations
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.rerankers.base import BaseReranker
from rag.schemas import RetrievalCandidate
from rag.vector_store import VectorStoreService
from safety.security import UnsafeInputError, assert_safe_retrieved_content
from services.cache import SemanticCache
from utils.config_handler import chroma_conf
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    return prompt


@dataclass
class EvidenceChunk:
    id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass
class Citation:
    evidence_id: str
    source: str


@dataclass
class RagResult:
    answer: str
    evidence: List[EvidenceChunk] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


def _build_reranker(cfg: Dict[str, Any]) -> Optional[BaseReranker]:
    if not cfg.get("enable_reranker"):
        return None
    try:
        from rag.rerankers.bge_reranker import BGEReranker
        return BGEReranker(model_name=cfg.get("reranker_model", "BAAI/bge-reranker-v2-m3"))
    except Exception:
        return None


class RagSummarizeService(object):
    def __init__(self, enable_semantic_cache: bool = True):
        self.vector_store_service = VectorStoreService()
        self.vector_store = self.vector_store_service.vector_store  # 保留字段兼容旧测试
        self._retrieval_cfg = chroma_conf.get("retrieval") or {}
        self._hybrid: Optional[HybridRetriever] = None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self._chain = None
        self._semantic_cache = None
        if enable_semantic_cache and embed_model is not None:
            try:
                self._semantic_cache = SemanticCache(
                    embedder=embed_model.embed_query,
                    threshold=0.92,
                    name="rag_semantic",
                )
            except Exception:
                self._semantic_cache = None

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._init_chain()
        return self._chain

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid is None:
            cfg = self._retrieval_cfg
            dense = DenseRetriever(self.vector_store_service.vector_store)
            bm25: Optional[BM25Retriever] = None
            if cfg.get("enable_bm25", True):
                try:
                    bm25 = self.vector_store_service.get_bm25_retriever()
                except Exception:
                    bm25 = None
            self._hybrid = HybridRetriever(
                dense=dense,
                bm25=bm25,
                reranker=_build_reranker(cfg),
                dense_k=int(cfg.get("dense_k", 20)),
                bm25_k=int(cfg.get("bm25_k", 20)),
                rrf_k=int(cfg.get("rrf_k", 60)),
                rerank_top_n=int(cfg.get("fusion_top_n", 20)),
                final_k=int(cfg.get("final_top_n", chroma_conf.get("k", 5))),
            )
        return self._hybrid

    def retrieve(self, query: str) -> List[RetrievalCandidate]:
        return self.hybrid_retriever.retrieve(query)

    def rag_summarize_result(self, query: str) -> RagResult:
        if self._semantic_cache is not None:
            cached = self._semantic_cache.get(query)
            if cached is not None:
                metrics_registry.inc_counter("agent_rag_cache_hit_total")
                return RagResult(answer=cached)

        candidates = self.retrieve(query)

        context = ""
        counter = 0
        evidence: List[EvidenceChunk] = []
        citations_structured: List[Citation] = []
        safe_docs = []
        for candidate in candidates:
            doc = candidate.document
            try:
                assert_safe_retrieved_content(doc.page_content)
            except UnsafeInputError:
                continue
            counter += 1
            evidence_id = candidate.doc_id
            source = candidate.source
            evidence.append(
                EvidenceChunk(
                    id=evidence_id,
                    source=source,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    score=candidate.final_score(),
                )
            )
            citations_structured.append(Citation(evidence_id=evidence_id, source=source))
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            safe_docs.append(doc)

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        citations = format_citations(safe_docs)
        result = f"{answer}\n\n引用来源：\n{citations}" if citations else answer
        if self._semantic_cache is not None:
            self._semantic_cache.set(query, result)
        return RagResult(answer=result, evidence=evidence, citations=citations_structured)

    def rag_summarize(self, query: str) -> str:
        return self.rag_summarize_result(query).answer


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
