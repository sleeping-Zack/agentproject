"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model, embed_model
from observability.metrics import metrics_registry
from rag.rag_utils import format_citations, hybrid_rank
from rag.vector_store import VectorStoreService
from safety.security import UnsafeInputError, assert_safe_retrieved_content
from services.cache import SemanticCache
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    return prompt


class RagSummarizeService(object):
    def __init__(self, enable_semantic_cache: bool = True):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
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

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        if self._semantic_cache is not None:
            cached = self._semantic_cache.get(query)
            if cached is not None:
                metrics_registry.inc_counter("agent_rag_cache_hit_total")
                return cached

        context_docs = hybrid_rank(
            query,
            self.retriever_docs(query),
            keyword_weight=0.35,
            top_n=None,
        )

        context = ""
        counter = 0
        for doc in context_docs:
            try:
                assert_safe_retrieved_content(doc.page_content)
            except UnsafeInputError:
                continue
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        citations = format_citations(context_docs)
        result = f"{answer}\n\n引用来源：\n{citations}" if citations else answer
        if self._semantic_cache is not None:
            self._semantic_cache.set(query, result)
        return result


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
