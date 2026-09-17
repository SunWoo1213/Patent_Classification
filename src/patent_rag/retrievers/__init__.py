"""Retriever 생성 진입점.

벡터DB가 디스크에 이미 있으면 그대로 불러오고, 비어 있을 때만 데이터를 읽어 새로 만든다.
"""
import logging
import shutil

from langchain_core.retrievers import BaseRetriever

from patent_rag import config
from patent_rag.data import load_documents
from patent_rag.models import get_embeddings
from patent_rag.retrievers.chunk import build_chunk_retriever
from patent_rag.retrievers.self_query import build_self_query_retriever
from patent_rag.retrievers.summary import build_summary_retriever
from patent_rag.vectorstore import is_empty, open_vectorstore

logger = logging.getLogger(__name__)

BUILDERS = {
    "summary": build_summary_retriever,
    "chunk": build_chunk_retriever,
    "meta": build_self_query_retriever,
}


def get_retriever(method: str, provider: str = config.EMBEDDING_PROVIDER,
                  rebuild: bool = False) -> BaseRetriever:
    if method not in BUILDERS:
        raise ValueError(f"지원하지 않는 방식입니다: {method} (선택: {', '.join(BUILDERS)})")
    # provider 는 저장 경로에 들어가므로 rmtree 전에 반드시 검증한다. (경로 조작 방지)
    if provider not in config.EMBEDDING_PROVIDERS:
        raise ValueError(f"지원하지 않는 임베딩 제공자입니다: {provider}")

    store_dir = config.store_dir(provider, method)
    if rebuild and store_dir.exists():
        logger.info("기존 저장소 삭제: %s", store_dir)
        shutil.rmtree(store_dir)

    vectorstore = open_vectorstore(config.COLLECTIONS[method], store_dir, get_embeddings(provider))
    docs = None
    if is_empty(vectorstore):
        logger.info("[%s] 저장된 벡터DB가 없어 새로 생성합니다.", method)
        docs = load_documents()
    else:
        logger.info("[%s] 저장된 벡터DB를 불러옵니다: %s", method, store_dir)
    return BUILDERS[method](vectorstore, store_dir, docs)


__all__ = ["BUILDERS", "get_retriever"]
