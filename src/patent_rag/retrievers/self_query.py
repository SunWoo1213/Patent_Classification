"""방법3: 메타데이터 기반 Self-Query Retriever (최종 채택).

LLM 이 자연어 질문을 '의미 검색어 + 메타데이터 필터' 로 분해하므로,
'전체 IPC' 처럼 정확히 일치해야 하는 값도 찾을 수 있다.
"""
from pathlib import Path

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from patent_rag.models import get_chat_model
from patent_rag.prompts import DOCUMENT_CONTENTS, METADATA_FIELDS
from patent_rag.vectorstore import add_documents

METADATA_FIELD_INFO = [
    AttributeInfo(name=name, description=description, type="string")
    for name, description in METADATA_FIELDS
]


def build_self_query_retriever(vectorstore: Chroma, store_dir: Path,
                               docs: list[Document] | None) -> SelfQueryRetriever:
    if docs:
        add_documents(vectorstore, docs)
    return SelfQueryRetriever.from_llm(
        llm=get_chat_model(),
        vectorstore=vectorstore,
        document_contents=DOCUMENT_CONTENTS,
        metadata_field_info=METADATA_FIELD_INFO,
    )
