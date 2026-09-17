"""방법2: Parent/Child Chunk Multi-Vector Retriever.

- 원문을 600자(부모 청크)와 300자(자식 청크)로 나누어 모두 벡터DB에 저장한다.
- 청크마다 메타데이터 문자열을 덧붙여, 잘린 조각에서도 원본 특허를 식별할 수 있게 한다.
- 검색 결과는 doc_id 로 연결된 원문 Document 로 반환한다.
"""
from pathlib import Path

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from patent_rag.data import ID_KEY, child_metadata, metadata_suffix
from patent_rag.vectorstore import add_documents

TOP_K = 10
PARENT_CHUNK_SIZE = 600
CHILD_CHUNK_SIZE = 300


def split_with_metadata(docs: list[Document], chunk_size: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    chunks = []
    for doc in docs:
        suffix = metadata_suffix(doc, include_id=True)
        for piece in splitter.split_documents([doc]):
            chunks.append(Document(page_content=piece.page_content + suffix,
                                   metadata=child_metadata(doc)))
    return chunks


def build_chunk_retriever(vectorstore: Chroma, store_dir: Path,
                          docs: list[Document] | None) -> MultiVectorRetriever:
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=LocalFileStore(store_dir / "docstore"),
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )
    if docs:
        add_documents(vectorstore, split_with_metadata(docs, PARENT_CHUNK_SIZE))
        add_documents(vectorstore, split_with_metadata(docs, CHILD_CHUNK_SIZE))
        retriever.docstore.mset([(doc.metadata[ID_KEY], doc) for doc in docs])
    return retriever
