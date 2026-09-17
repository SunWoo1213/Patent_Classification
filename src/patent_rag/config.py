"""프로젝트 전역 설정.

모든 값은 환경변수 또는 프로젝트 루트의 `.env` 파일로 덮어쓸 수 있다.
API 키는 코드에 직접 적지 않고 반드시 환경변수(OPENAI_API_KEY)로 전달한다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: Path) -> Path:
    return Path(os.getenv(env_name, default))


# --- 데이터 ---
DATA_DIR = _path("DATA_DIR", PROJECT_ROOT / "data")
TRAIN_FILE = DATA_DIR / os.getenv("TRAIN_FILE", "DS학술제-모델링경진대회_Train.xlsx")
VALID_FILE = DATA_DIR / os.getenv("VALID_FILE", "DS학술제-모델링경진대회_Valid.xlsx")

# --- 저장소 (벡터DB, 부모 문서, 요약 캐시) ---
STORE_DIR = _path("STORE_DIR", PROJECT_ROOT / "store")
SUMMARY_CACHE = _path("SUMMARY_CACHE", STORE_DIR / "summaries.json")

# --- 임베딩 ---
EMBEDDING_PROVIDERS = ("huggingface", "openai")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # GPU 환경에서는 cuda

# --- LLM ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-turbo-preview")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

# --- Retriever 방식별 컬렉션 이름 ---
COLLECTIONS = {
    "summary": "summarise",  # 방법1: 요약 기반 Multi-Vector
    "chunk": "smallize",     # 방법2: Parent/Child Chunk Multi-Vector
    "meta": "meta",          # 방법3: 메타데이터 기반 Self-Query
}
METHODS = tuple(COLLECTIONS)


def store_dir(provider: str, method: str) -> Path:
    """임베딩 모델마다 벡터 차원이 다르므로 저장 경로를 임베딩 제공자별로 분리한다."""
    return STORE_DIR / provider / method


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일 또는 환경변수로 설정하세요."
        )
