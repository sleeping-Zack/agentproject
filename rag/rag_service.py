"""
总结服务类：用户提问，走 Hybrid 检索（Dense + BM25 + RRF + 可选 Rerank），把证据交给模型总结。
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent.verifier import AnswerVerifier
from model.factory import chat_model, embed_model
from observability.context import request_context
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
from utils.config_handler import chroma_conf, rag_conf
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
    verification: Optional[Dict[str, Any]] = None


def _build_reranker(cfg: Dict[str, Any]) -> Optional[BaseReranker]:
    if not cfg.get("enable_reranker"):
        return None
    try:
        from rag.rerankers.bge_reranker import BGEReranker
        return BGEReranker(model_name=cfg.get("reranker_model", "BAAI/bge-reranker-v2-m3"))
    except Exception:
        return None


class RagSummarizeService(object):
    def __init__(
        self,
        enable_semantic_cache: bool = True,
        verifier: Optional[AnswerVerifier] = None,
        verify_generation: bool = True,
    ):
        self.vector_store_service = VectorStoreService()
        self.vector_store = self.vector_store_service.vector_store  # 保留字段兼容旧测试
        self._retrieval_cfg = chroma_conf.get("retrieval") or {}
        self._hybrid: Optional[HybridRetriever] = None
        self.prompt_text = load_rag_prompts()
        self._prompt_version = request_context().prompt_version or "rag_summarize:unversioned"
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.verifier = verifier or AnswerVerifier()
        self.verify_generation = verify_generation
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

    def _semantic_cache_namespace(
        self,
        *,
        tenant_id: Optional[str],
        knowledge_base_id: Optional[str],
        corpus_version: Optional[str],
        prompt_version: Optional[str],
        retrieval_version: Optional[str],
        model_version: Optional[str],
    ) -> Dict[str, str]:
        ctx = request_context()
        extra = ctx.extra or {}
        retrieval_cfg = getattr(self, "_retrieval_cfg", {}) or {}
        configured_model = (
            f"{rag_conf.get('model_provider', 'unknown')}:"
            f"{rag_conf.get('chat_model_name', 'unknown')}"
        )
        return {
            "tenant_id": str(tenant_id or ctx.tenant_id or "default"),
            "knowledge_base_id": str(
                knowledge_base_id
                or extra.get("knowledge_base_id")
                or chroma_conf.get("knowledge_base_id")
                or chroma_conf.get("collection_name")
                or "default"
            ),
            "corpus_version": str(
                corpus_version
                or extra.get("corpus_version")
                or chroma_conf.get("corpus_version")
                or chroma_conf.get("chunk_version")
                or "unversioned"
            ),
            "prompt_version": str(
                prompt_version
                or extra.get("rag_prompt_version")
                or getattr(self, "_prompt_version", None)
                or ctx.prompt_version
                or "unversioned"
            ),
            "retrieval_version": str(
                retrieval_version
                or extra.get("retrieval_version")
                or retrieval_cfg.get("version")
                or "unversioned"
            ),
            "model_version": str(
                model_version
                or ctx.model
                or extra.get("model_version")
                or configured_model
            ),
        }

    def rag_summarize_result(
        self,
        query: str,
        *,
        tenant_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        corpus_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
        retrieval_version: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> RagResult:
        cache_namespace = None
        if self._semantic_cache is not None:
            cache_namespace = self._semantic_cache_namespace(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                corpus_version=corpus_version,
                prompt_version=prompt_version,
                retrieval_version=retrieval_version,
                model_version=model_version,
            )
            cached = self._semantic_cache.get(query, namespace=cache_namespace)
            if isinstance(cached, RagResult):
                metrics_registry.inc_counter("agent_rag_cache_hit_total")
                return deepcopy(cached)
            if isinstance(cached, str):
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

        if not evidence:
            result = RagResult(
                answer="请求未执行：知识库中没有足够证据支持回答该问题。",
                evidence=[],
                citations=[],
                verification={
                    "passed": False,
                    "action": "refuse",
                    "reasons": ["evidence_required"],
                },
            )
            if self._semantic_cache is not None:
                self._semantic_cache.set(query, deepcopy(result), namespace=cache_namespace)
            return result

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        citations = format_citations(safe_docs)
        answer_with_citations = f"{answer}\n\n引用来源：\n{citations}" if citations else answer
        verification = None
        if getattr(self, "verify_generation", False):
            verifier = getattr(self, "verifier", None) or AnswerVerifier()
            verified = verifier.verify(
                query=query,
                answer=answer_with_citations,
                evidence=[item.__dict__ for item in evidence],
                scene="rag",
            )
            verification = {
                "passed": verified.passed,
                "action": verified.action,
                "score": verified.score,
                "reasons": list(verified.reasons),
                **verified.quality,
            }
            if not verified.passed:
                answer_with_citations = (
                    "请求未执行：生成结果未通过证据一致性校验，"
                    "知识库中没有足够证据支持该结论。"
                )
        result = RagResult(
            answer=answer_with_citations,
            evidence=evidence,
            citations=citations_structured,
            verification=verification,
        )
        if self._semantic_cache is not None:
            self._semantic_cache.set(query, deepcopy(result), namespace=cache_namespace)
        return result

    def rag_summarize(self, query: str) -> str:
        return self.rag_summarize_result(query).answer


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
