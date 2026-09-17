"""특허 엑셀 데이터를 불러와 정제하고 LangChain Document 로 변환한다."""
import logging
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from patent_rag import config

logger = logging.getLogger(__name__)

ID_KEY = "doc_id"

# 검색 필터 및 청크 식별에 사용하는 메타데이터 컬럼
METADATA_COLUMNS = ["메인IPC2", "전체 IPC", "출원번호", "등록번호", "출원인", "발명자", "법적상태"]

# 요약/청크(자식) 문서에 남겨둘 최소 메타데이터
CHILD_METADATA_COLUMNS = ["메인IPC2", "전체 IPC", "출원번호"]


def load_patents(train_file: Path = config.TRAIN_FILE,
                 valid_file: Path = config.VALID_FILE) -> pd.DataFrame:
    """학습용·검증용 데이터를 합치고 중복 행을 제거한다."""
    for f in (train_file, valid_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {f}")

    frames = [pd.read_excel(f) for f in (train_file, valid_file)]
    for f, df in zip((train_file, valid_file), frames):
        logger.info("%s: %d건", Path(f).name, len(df))

    data = pd.concat(frames, ignore_index=True)
    if data.isnull().values.any():
        logger.warning("결측치가 존재합니다. 결측값은 문자열 'nan' 으로 저장됩니다.")

    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    logger.info("중복 제거: %d건 → %d건", before, len(data))
    return data


def make_doc_id(page_content: str) -> str:
    """문서 내용으로부터 항상 같은 UUID 를 만든다.

    실행할 때마다 임의의 UUID 를 만들면 디스크에 저장한 벡터DB와
    부모 문서 저장소의 키가 어긋나므로, 내용 기반 UUID5 를 사용한다.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, page_content))


def to_documents(data: pd.DataFrame) -> list[Document]:
    """DataFrame 의 각 행을 '컬럼 : 값' 텍스트와 메타데이터를 가진 Document 로 변환한다."""
    docs = []
    for _, row in data.iterrows():
        content = " ;".join(f"{col} : {row[col]}" for col in data.columns)
        metadata = {col: str(row[col]) for col in METADATA_COLUMNS}
        metadata[ID_KEY] = make_doc_id(content)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_documents() -> list[Document]:
    return to_documents(load_patents())


def metadata_suffix(doc: Document, include_id: bool = False) -> str:
    """잘리거나 요약된 텍스트에서도 원본 특허를 식별할 수 있도록 덧붙일 메타 문자열."""
    parts = [f"{col} : {doc.metadata[col]}" for col in METADATA_COLUMNS]
    if include_id:
        parts.append(f"{ID_KEY} : {doc.metadata[ID_KEY]}")
    return "; " + "; ".join(parts)


def child_metadata(doc: Document) -> dict:
    metadata = {col: doc.metadata[col] for col in CHILD_METADATA_COLUMNS}
    metadata[ID_KEY] = doc.metadata[ID_KEY]
    return metadata
