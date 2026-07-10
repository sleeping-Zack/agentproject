from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from rag.rag_utils import build_document_metadata
from rag.retrievers.bm25_retriever import BM25Retriever
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self._bm25: BM25Retriever | None = None

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _bm25_index_path(self) -> str | None:
        rel = chroma_conf.get("bm25_index_path")
        if not rel:
            return None
        return get_abs_path(rel)

    def _all_documents_from_chroma(self) -> list[Document]:
        """从 Chroma 里 dump 所有文档，用于现场重建 BM25 索引。"""
        try:
            payload = self.vector_store.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning(f"[BM25]从 Chroma dump 文档失败：{exc}")
            return []
        docs: list[Document] = []
        for content, meta in zip(payload.get("documents") or [], payload.get("metadatas") or []):
            if content is None:
                continue
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def get_bm25_retriever(self) -> BM25Retriever:
        """惰性获取 BM25 索引。加载失败或与 Chroma 数量不一致就重建。"""
        if self._bm25 is not None and self._bm25.is_ready():
            return self._bm25

        bm25 = BM25Retriever(index_path=self._bm25_index_path())
        loaded = bm25.load()
        chroma_docs = self._all_documents_from_chroma()

        if loaded:
            expected = len(chroma_docs)
            actual = len(bm25._payload.doc_ids) if bm25._payload else 0
            if expected and expected != actual:
                logger.warning(
                    f"[BM25]索引与 Chroma 数量不一致 (chroma={expected}, bm25={actual})，重建"
                )
                loaded = False

        if not loaded and chroma_docs:
            bm25.build(chroma_docs)
            bm25.save()
            logger.info(f"[BM25]索引从 Chroma 重建完成，文档数={len(chroma_docs)}")

        self._bm25 = bm25
        return bm25

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            os.makedirs(os.path.dirname(get_abs_path(chroma_conf["md5_hex_store"])), exist_ok=True)
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        any_added = False
        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                base_metadata = build_document_metadata(
                    path,
                    chunk_version=chroma_conf.get("chunk_version", "v1"),
                )
                for index, doc in enumerate(split_document):
                    doc.metadata.update(base_metadata)
                    doc.metadata["chunk_index"] = index
                    doc.metadata["doc_id"] = f"{md5_hex}:{index}"

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)
                any_added = True

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

        if any_added:
            # 增量入库后强制重建 BM25 索引，保持与 Chroma 一致
            self._bm25 = None
            try:
                self.get_bm25_retriever()
            except Exception as exc:
                logger.warning(f"[BM25]重建失败：{exc}")


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)
