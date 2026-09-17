"""Chroma 벡터DB 생성·조회 유틸리티."""
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000  # Chroma 의 1회 최대 입력 건수 제한을 피하기 위해 나누어 저장


def open_vectorstore(collection_name: str, persist_dir: Path, embeddings: Embeddings) -> Chroma:
    """디스크에 저장된 컬렉션이 있으면 불러오고, 없으면 빈 컬렉션을 만든다."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir / "chroma"),
        embedding_function=embeddings,
    )


def is_empty(vectorstore: Chroma) -> bool:
    return not vectorstore.get(limit=1)["ids"]


def add_documents(vectorstore: Chroma, docs: list[Document]) -> None:
    for start in range(0, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[start:start + BATCH_SIZE])
        logger.info("벡터DB 저장 중: %d / %d", min(start + BATCH_SIZE, len(docs)), len(docs))
