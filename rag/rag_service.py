"""
总结服务类：用户提问，走 Hybrid 检索（Dense + BM25 + RRF + 可选 Rerank），把证据交给模型总结。
"""
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate

from agent.budget import BudgetManager, current_budget_manager
from agent.budgeted_text_model import invoke_budgeted_call, model_response_text
from agent.verifier import AnswerVerifier, build_default_answer_verifier
from model.factory import chat_model, embed_model
from observability.context import request_context
from observability.metrics import metrics_registry
from rag.rag_utils import format_citations, knowledge_gap_answer
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.rerankers.base import BaseReranker
from rag.rerankers.factory import build_reranker
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
    business_status: str = "success"


def _build_reranker(cfg: Dict[str, Any]) -> Optional[BaseReranker]:
    return build_reranker(cfg)


class RagSummarizeService(object):
    def __init__(
        self,
        enable_semantic_cache: bool = True,
        verifier: Optional[AnswerVerifier] = None,
        verify_generation: bool = True,
        budget_manager: Optional[BudgetManager] = None,
    ):
        self.vector_store_service = VectorStoreService()
        self.vector_store = self.vector_store_service.vector_store  # 保留字段兼容旧测试
        self._retrieval_cfg = chroma_conf.get("retrieval") or {}
        self._hybrid: Optional[HybridRetriever] = None
        self.prompt_text = load_rag_prompts()
        self._prompt_version = request_context().prompt_version or "rag_summarize:unversioned"
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.verifier = verifier or build_default_answer_verifier()
        self.verify_generation = verify_generation
        self.budget_manager = budget_manager
        self.max_output_tokens = max(
            1,
            int(os.getenv("AGENT_RAG_MAX_OUTPUT_TOKENS", "1600")),
        )
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

    def _init_chain(self, max_output_tokens: Optional[int] = None):
        model = self.model.resolve() if hasattr(self.model, "resolve") else self.model
        output_limit = max_output_tokens or self.max_output_tokens
        if hasattr(model, "bind"):
            model = model.bind(max_tokens=max(1, int(output_limit)))
        chain = self.prompt_template | model
        return chain

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._init_chain(self.max_output_tokens)
        return self._chain

    def _generation_chain(self, max_output_tokens: int):
        if self._chain is not None:
            return self._chain
        return self._init_chain(max_output_tokens=max_output_tokens)

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
                rerank_strategy=str(cfg.get("rerank_strategy", "shadow")),
                rerank_hybrid_weight=float(cfg.get("rerank_hybrid_weight", 0.7)),
                rerank_model_weight=float(cfg.get("rerank_model_weight", 0.3)),
                rerank_fusion_k=int(cfg.get("rerank_fusion_k", 10)),
                rerank_bypass_exact_queries=bool(
                    cfg.get("rerank_bypass_exact_queries", True)
                ),
                fusion_anchor_k=int(cfg.get("fusion_anchor_k", 20)),
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
        configured_retrieval_version = str(
            retrieval_cfg.get("version") or "unversioned"
        )
        if retrieval_cfg.get("enable_reranker"):
            configured_retrieval_version = (
                f"{configured_retrieval_version}:"
                f"{retrieval_cfg.get('rerank_version', 'rerank-unversioned')}"
            )
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
                or configured_retrieval_version
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
        budget_manager: Optional[BudgetManager] = None,
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

        dense_relevance = max(
            (candidate.dense_score for candidate in candidates if candidate.dense_score is not None),
            default=0.0,
        )
        sparse_relevance = max(
            (candidate.sparse_score for candidate in candidates if candidate.sparse_score is not None),
            default=0.0,
        )
        retrieval_supported = (
            dense_relevance >= float(self._retrieval_cfg.get("min_dense_relevance", 0.15))
            or sparse_relevance >= float(self._retrieval_cfg.get("min_sparse_relevance", 1.0))
        )

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
            context += (
                f"【参考资料{counter}｜证据ID:{evidence_id}】: "
                f"参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            )
            safe_docs.append(doc)

        if not evidence or not retrieval_supported:
            reason = "evidence_required" if not evidence else "retrieval_relevance_below_threshold"
            result = RagResult(
                answer=knowledge_gap_answer(query, reason),
                evidence=[],
                citations=[],
                business_status="empty",
                verification={
                    "passed": False,
                    "action": "refuse",
                    "reasons": [reason],
                    "dense_relevance": dense_relevance,
                    "sparse_relevance": sparse_relevance,
                },
            )
            if self._semantic_cache is not None:
                self._semantic_cache.set(query, deepcopy(result), namespace=cache_namespace)
            return result

        citations = format_citations(
            safe_docs,
            evidence_ids=[item.id for item in evidence],
        )
        verification = None
        answer_with_citations = ""
        generation_input = query
        manager = self._resolve_budget_manager(budget_manager)
        max_attempts = 2 if getattr(self, "verify_generation", False) else 1
        for attempt in range(max_attempts):
            answer = self._generate(
                {
                    "input": generation_input,
                    "context": context,
                },
                budget_manager=manager,
                attempt=attempt,
            )
            answer_with_citations = (
                f"{answer}\n\n引用来源：\n{citations}" if citations else answer
            )
            if not getattr(self, "verify_generation", False):
                break
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
            if verified.passed:
                break
            if attempt + 1 < max_attempts:
                generation_input = self._retry_input(
                    query=query,
                    previous_answer=answer_with_citations,
                    verification=verification,
                )
        if verification is not None and not verification["passed"]:
            reasons = list(verification.get("reasons") or [])
            unsupported_rate = float(verification.get("unsupported_claim_rate") or 0.0)
            # 仅在"真正的未落地/有害"时拒绝：有害指令、证据矛盾，或几乎所有声明都无证据支撑。
            # 其余（如弱语义支撑判断误伤、但 citation_validity/coverage 已达标的可用摘要）保留摘要，
            # 避免"丢弃摘要→降级裸原文→下游引用闸门失败→retry→token 护栏→max_tokens_exceeded"的级联。
            hard_fail = (
                "harmful_instruction" in reasons
                or "evidence_contradiction" in reasons
                or unsupported_rate >= 1.0
            )
            if hard_fail:
                answer_with_citations = (
                    "请求未执行：生成结果未通过证据一致性校验，"
                    "知识库中没有足够证据支持该结论。"
                )
        result = RagResult(
            answer=answer_with_citations,
            evidence=evidence,
            citations=citations_structured,
            verification=verification,
            business_status=(
                "verification_failed"
                if verification is not None and not verification["passed"]
                else "success"
            ),
        )
        if self._semantic_cache is not None:
            self._semantic_cache.set(query, deepcopy(result), namespace=cache_namespace)
        return result

    def rag_summarize(
        self,
        query: str,
        *,
        budget_manager: Optional[BudgetManager] = None,
    ) -> str:
        return self.rag_summarize_result(
            query,
            budget_manager=budget_manager,
        ).answer

    def _resolve_budget_manager(
        self,
        budget_manager: Optional[BudgetManager],
    ) -> Optional[BudgetManager]:
        if budget_manager is not None:
            return budget_manager
        configured = getattr(self, "budget_manager", None)
        return configured if configured is not None else current_budget_manager()

    def _generate(
        self,
        payload: Dict[str, str],
        *,
        budget_manager: Optional[BudgetManager],
        attempt: int,
    ) -> str:
        prompt = "\n".join(
            (
                str(getattr(self, "prompt_text", "")),
                str(payload.get("input") or ""),
                str(payload.get("context") or ""),
            )
        )
        max_output_tokens = int(
            getattr(
                self,
                "max_output_tokens",
                os.getenv("AGENT_RAG_MAX_OUTPUT_TOKENS", "1600"),
            )
        )
        response = invoke_budgeted_call(
            lambda output_cap: self._generation_chain(output_cap).invoke(payload),
            prompt,
            max_output_tokens=max_output_tokens,
            operation=f"rag-generation-{attempt + 1}",
            budget_manager=budget_manager,
            model_name=type(getattr(self, "model", None)).__name__,
            retry=attempt,
        )
        return model_response_text(response)

    @staticmethod
    def _retry_input(
        *,
        query: str,
        previous_answer: str,
        verification: Dict[str, Any],
    ) -> str:
        reasons = ", ".join(str(item) for item in verification.get("reasons") or ())
        return (
            f"{query}\n\n"
            "上一版答案未通过证据一致性校验。请只依据参考资料修订，不得重复原答案中"
            "缺少证据的结论；保留有证据支持的内容并给出准确引用。\n"
            f"上一版答案：\n{previous_answer}\n"
            f"校验反馈：{reasons or verification.get('action') or 'verification_failed'}"
        )

if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
