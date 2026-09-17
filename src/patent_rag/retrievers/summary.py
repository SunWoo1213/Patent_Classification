"""방법1: LLM 요약 기반 Multi-Vector Retriever.

- 자식(검색용): 문서 요약 + 핵심 메타데이터 → 벡터DB
- 부모(반환용): 원문 Document → LocalFileStore
"""
import json
import logging
from pathlib import Path

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from patent_rag import config
from patent_rag.data import ID_KEY, child_metadata, metadata_suffix
from patent_rag.models import get_chat_model
from patent_rag.prompts import SUMMARY_PROMPT
from patent_rag.vectorstore import add_documents

logger = logging.getLogger(__name__)

TOP_K = 3


def summarize_documents(docs: list[Document], cache_path: Path = config.SUMMARY_CACHE,
                        max_concurrency: int = 5) -> list[str]:
    """문서별 3문장 요약을 만든다. 결과는 캐시에 저장해 API 비용이 반복되지 않게 한다."""
    # pickle 은 로드 시 임의 코드가 실행될 수 있으므로 캐시는 JSON 으로 저장한다.
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            summaries = json.load(f)
        if len(summaries) == len(docs):
            logger.info("요약 캐시 사용: %s", cache_path)
            return summaries
        logger.warning("요약 캐시 건수(%d)가 문서 수(%d)와 달라 다시 생성합니다.", len(summaries), len(docs))

    chain = (
        RunnableLambda(lambda doc: {"doc": doc.page_content})
        | SUMMARY_PROMPT
        | get_chat_model(config.SUMMARY_MODEL)
        | StrOutputParser()
    )
    logger.info("문서 %d건 요약 생성 중 (동시 처리 %d)", len(docs), max_concurrency)
    summaries = chain.batch(docs, {"max_concurrency": max_concurrency})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False)
    return summaries


def build_summary_retriever(vectorstore: Chroma, store_dir: Path,
                            docs: list[Document] | None) -> MultiVectorRetriever:
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=LocalFileStore(store_dir / "docstore"),
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )
    if docs:
        summaries = summarize_documents(docs)
        sub_docs = [
            Document(page_content=summary + metadata_suffix(doc), metadata=child_metadata(doc))
            for summary, doc in zip(summaries, docs)
        ]
        add_documents(vectorstore, sub_docs)
        retriever.docstore.mset([(doc.metadata[ID_KEY], doc) for doc in docs])
    return retriever
