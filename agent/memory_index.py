from __future__ import annotations

from typing import List, Protocol, Sequence

from agent.long_term_memory import MemoryRecord


class MemorySearchIndex(Protocol):
    def upsert(self, memory: MemoryRecord) -> None: ...
    def delete(self, memory_ids: Sequence[str]) -> None: ...
    def query(self, tenant_id: str, user_id: str, text: str, limit: int) -> List[str]: ...


class ChromaMemoryIndex:
    """Rebuildable vector candidate index; relational storage remains authoritative."""

    def __init__(
        self,
        persist_directory: str,
        embedding_model,
        collection_name: str = "agent_user_memory",
    ) -> None:
        import chromadb

        self.embedding_model = embedding_model
        client = chromadb.PersistentClient(path=persist_directory)
        self.collection = client.get_or_create_collection(collection_name)

    def upsert(self, memory: MemoryRecord) -> None:
        document = f"{memory.key}: {memory.value}"
        embedding = self.embedding_model.embed_query(document)
        self.collection.upsert(
            ids=[memory.memory_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[
                {
                    "tenant_id": memory.tenant_id,
                    "user_id": memory.user_id,
                    "category": memory.category.value,
                }
            ],
        )

    def delete(self, memory_ids: Sequence[str]) -> None:
        if memory_ids:
            self.collection.delete(ids=list(memory_ids))

    def query(self, tenant_id: str, user_id: str, text: str, limit: int) -> List[str]:
        embedding = self.embedding_model.embed_query(text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, limit),
            where={
                "$and": [
                    {"tenant_id": {"$eq": tenant_id}},
                    {"user_id": {"$eq": user_id}},
                ]
            },
            include=["distances"],
        )
        ids = result.get("ids") or []
        return list(ids[0]) if ids else []
