"""임베딩 모델과 LLM 생성 함수."""
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from patent_rag import config


def get_embeddings(provider: str = config.EMBEDDING_PROVIDER) -> Embeddings:
    if provider == "huggingface":
        # KorNLU 데이터셋으로 학습한 한국어 문장 임베딩 모델
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=config.HF_EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": False},
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        config.require_openai_key()
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    raise ValueError(f"지원하지 않는 임베딩 제공자입니다: {provider}")


def get_chat_model(model: str = config.CHAT_MODEL, temperature: float = 0) -> ChatOpenAI:
    config.require_openai_key()
    return ChatOpenAI(model=model, temperature=temperature)
