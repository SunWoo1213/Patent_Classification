# 특허 검색 · IPC 코드 생성 RAG 시스템

특허 3,006건을 벡터DB에 넣고, 질문 의도에 따라 기존 특허 조회와 새 발명의 IPC 코드 생성을 하나의 질의 함수로 처리하는 시스템입니다.

<p>
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-412991?logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00"/>
  <img src="https://img.shields.io/badge/HuggingFace-ko--sroberta-FFD21E?logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white"/>
</p>

특허 문서에는 성격이 다른 두 종류의 값이 섞여 있습니다. 발명의 명칭이나 요약, 대표청구항은 긴 글이라 의미로 찾아야 합니다. 반면 출원번호와 전체 IPC, 출원인, 법적상태는 글자 단위로 정확히 맞아야 합니다. 이 시스템은 두 조건을 나눠 입력하지 않아도 한 질의에서 처리합니다. 여기에 아직 출원하지 않은 발명의 설명을 넣으면 IPC 코드를 만들어 주는 경로를 붙였습니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 기간 | 2024.09 ~ 2024.11 |
| 인원과 본인 역할 | 2인 팀. 본인은 검색과 생성 로직 코드 전체를 맡았고, 팀원은 데이터 라벨링과 데이터 파악을 맡았습니다 |
| 상태 | 완료 (Colab 노트북 1개, 출력 제거본) |
| 수상 | 강남대학교 데이터사이언스전공 DS학술제 모델링 경진대회 장려상 (2024.11) |
| 핵심 기술 | LangChain 0.3 (`SelfQueryRetriever` · `MultiVectorRetriever` · LCEL `RunnableBranch`), ChromaDB, `jhgan/ko-sroberta-multitask`, OpenAI `gpt-4-turbo-preview` · `gpt-4o-mini` |

## 문제 해결 사례

1. [IPC 코드처럼 정확히 일치해야 하는 값을 임베딩 유사도로 못 찾던 문제를 메타데이터 필터로 해결](#1-ipc-코드처럼-정확히-일치해야-하는-값을-임베딩-유사도로-못-찾던-문제를-메타데이터-필터로-해결)
2. [Agent가 조회와 생성 경로를 일정하게 고르지 못하던 문제를 라우터 체인으로 해결](#2-agent가-조회와-생성-경로를-일정하게-고르지-못하던-문제를-라우터-체인으로-해결)
3. [재실행마다 요약 API를 다시 부르던 문제를 캐싱으로 해결](#3-재실행마다-요약-api를-다시-부르던-문제를-캐싱으로-해결)

### 1. IPC 코드처럼 정확히 일치해야 하는 값을 임베딩 유사도로 못 찾던 문제를 메타데이터 필터로 해결

**문제** "전체 IPC가 `G16H-010/60,[G06Q-010/10, G16H-010/20, G16H-080/00]`와 정확히 일치하는 특허"를 찾는 질의에서, `MultiVectorRetriever` 기반 방법 1(요약)과 방법 2(청크)는 비슷한 문서만 돌려줄 뿐 전체 IPC가 같은지는 보장하지 못했습니다.

**원인**
- 임베딩 텍스트 뒤에 붙인 IPC 문자열은 벡터 하나에서 작은 비중이라, `G16H-010/60`과 `G16H-010/20`의 차이가 벡터 거리에 드러나기 어렵습니다.
- 유사도 검색은 일치하는 문서가 없어도 가장 가까운 k개를 항상 돌려줍니다.
- 방법 1 · 2에는 검색할 때 메타데이터 필터를 거는 단계가 없었습니다.

**해결**
- 방법 3에서 22개 컬럼 전체를 본문으로, 8개 필드(`doc_id` · `메인IPC2` · `전체 IPC` · 출원번호 · 등록번호 · 출원인 · 발명자 · 법적상태)를 metadata로 넣었습니다.
- `SelfQueryRetriever`가 질문을 의미 검색어와 구조화된 필터로 나누고, 필터는 Chroma의 `where` 조건으로 바뀌어 값이 같은 문서만 후보로 남깁니다.
- `AttributeInfo`에 필드 설명을 쓰고, IPC가 들어온 질문이면 `전체 IPC`로 정확 일치 필터를 만들라고 적었습니다.

**결과** 다섯 유형의 공통 테스트 질의로 세 Retriever를 비교했고, 방법 3에서 긴 IPC 문자열이 메타데이터 필터로 바뀌어 전체 IPC가 정확히 일치하는 특허가 반환됐습니다.

### 2. Agent가 조회와 생성 경로를 일정하게 고르지 못하던 문제를 라우터 체인으로 해결

**문제** Retriever를 Tool로 넘긴 OpenAI Agent가 같은 종류의 질문에도 조회로 답할 때와 LLM이 직접 답을 지어낼 때가 섞였습니다. 생성 경로는 실제 데이터에 없는 출원번호와 IPC를 만들어 낼 수 있습니다.

**원인** 프롬프트를 두 차례 고쳐도 Tool을 부를지는 매 질문마다 LLM이 정했고, 프롬프트에 "정보가 없으면 직접 생성"이라는 탈출구가 열려 있었습니다.

**해결**
- LLM의 역할을 라벨 하나(`search` · `generator` · `default`)만 출력하는 `router_chain`으로 줄였습니다.
- `RunnableBranch`가 라벨을 소문자 부분 문자열로 비교해 체인을 고르므로, 대소문자나 문장부호가 섞여도 같은 분기로 갑니다.
- `search` 체인은 `retriever03`(Self-Query) 호출이 코드에 고정돼 검색을 건너뛸 수 없고, `generator` 체인은 IPC 코드를 최대 10개 부여 사유와 함께 만듭니다.

**결과** 조회 질문은 `search`로 가서 항상 Retriever를 거치고, 생성 요청은 `generator`로 갑니다. 최종 체인에 네 가지 질의를 넣어 의도한 경로와 출력 양식을 확인했습니다.

### 3. 재실행마다 요약 API를 다시 부르던 문제를 캐싱으로 해결

**문제** Colab 런타임이 끊길 때마다 노트북을 처음부터 다시 돌려야 했고, 그때마다 특허 건수만큼 `gpt-4o-mini` 요약 비용과 대기 시간이 들어 Retriever 3종 비교를 반복하기 어려웠습니다.

**원인** 가장 비싼 중간 산출물인 요약 결과와 벡터DB가 메모리에만 있었고, 적재 셀을 다시 실행하면 Chroma 컬렉션에 같은 문서가 한 번 더 추가됐습니다.

**해결**
- 요약을 `batch(docs, {"max_concurrency": 5})`로 병렬 실행하고, 건수를 확인한 뒤 `list.pickle`로 Drive에 저장했습니다.
- 방법마다 Chroma `persist_directory`와 컬렉션 이름을 따로 두고 원문 저장소도 `LocalFileStore`로 디스크에 남겼습니다.
- 생성 셀 제목에 "벡터DB가 저장되어 있다면 SKIP"을 적어 저장본이 있으면 불러오기 셀만 실행하게 했습니다.

**결과** 재실행할 때 요약 API를 다시 부르지 않고, 저장된 벡터DB로 Retriever 비교 실험을 같은 요약 조건에서 이어갈 수 있게 됐습니다.

## 아키텍처

```mermaid
flowchart TD
    A[Train.xlsx + Valid.xlsx] --> B[병합 · 결측치 점검 · 중복 제거]
    B --> C[행 단위 Document 변환<br/>메타데이터 8개 · UUID 부여]
    C --> D{Retriever 전략 비교<br/>공통 테스트 질의}

    D --> E1[방법1 요약 기반<br/>Multi-Vector]
    D --> E2[방법2 600자·300자 청크<br/>Multi-Vector]
    D --> E3[방법3 메타데이터 기반<br/>Self-Query · 채택]

    E1 -.->|검증| AG[Retriever Tool + Agent]
    E2 -.->|검증| AG

    E3 --> F[Router Chain<br/>GPT-4 Turbo]
    U([사용자 질문]) --> F
    F -->|search| G[검색 체인<br/>질의 보강 → Retriever → 양식화된 답변]
    F -->|generator| H[IPC 생성 체인<br/>LLM 단독]
    F -->|default| I[일반 LLM 체인]
```

라우터가 질문을 `search` · `generator` · `default` 중 하나로 분류하면, 코드가 그 이름대로 체인을 나눕니다. 검색 체인은 Self-Query Retriever로 문서를 찾고, 생성 체인은 검색 없이 LLM이 IPC 코드와 부여 사유를 만듭니다.

## 기술 스택

| 구분 | 기술 | 선택한 이유 |
| --- | --- | --- |
| 언어 · 데이터 | Python, pandas, openpyxl | 대회 데이터가 엑셀 형식이었습니다 |
| 프레임워크 | LangChain 0.3 (LCEL, Agents, Retrievers, `RunnableBranch`) | Retriever 여러 종을 같은 인터페이스로 바꿔 끼우며 비교하려고 썼습니다 |
| 벡터 저장 | ChromaDB (디스크 영구 저장), `LocalFileStore` | Colab 런타임이 끊겨도 Drive에 남은 벡터DB를 다시 불러 쓰려고 썼습니다 |
| 모델 | OpenAI `gpt-4-turbo-preview` (Agent · Self-Query · Router · 생성), `gpt-4o-mini` (문서 요약) | 요약은 건수가 많아 저렴한 모델로, 질의 분해와 생성은 상위 모델로 나눴습니다 |
| 임베딩 | `jhgan/ko-sroberta-multitask` | 특허 원문이 한국어라 한국어 문장 임베딩을 썼습니다 |
| 환경 | Google Colab + Google Drive | |

## 역할과 기여도

2인 팀에서 검색과 생성 로직 코드 전체를 담당했습니다. 데이터 라벨링과 데이터 파악은 팀원이 맡았습니다.

- 특허 3,006건(22개 컬럼)을 행 단위 Document로 바꾸고, 메타데이터 8개 필드와 UUID를 설계했습니다.
- Retriever 3종을 직접 구현하고 같은 테스트 질의 세트로 비교해 방식을 정했습니다.
- `AttributeInfo`로 각 필드의 뜻을 기술해 LLM이 질문을 의미 검색어와 필터로 분해하게 했습니다.
- 방법 1 · 2에서는 청크마다 IPC · 출원번호 메타 문자열을 붙이고 UUID로 원문과 연결했습니다. 어느 조각이 검색되든 원문 전체를 돌려주도록 `MultiVectorRetriever`를 직접 구성했습니다.
- Agent를 `RunnableBranch` 라우터 체인으로 바꿔, 조회로 분류된 질문은 반드시 Retriever를 거치게 했습니다.
- IPC 생성 체인이 코드마다 한글 분류명과 부여 사유를 함께 내도록 프롬프트에 출력 양식을 지정했습니다.
- 요약 병렬 처리와 pickle 캐싱, 방법별 벡터DB의 Drive 영구 저장으로 재실행 때 요약 API 호출을 없앴습니다.

## 결과

원본 실행 로그에서 발췌했습니다. 출원번호 · 등록번호 · 출원인은 가렸고, 대회 데이터는 저장소에 넣지 않았습니다.

| 질의 | 경로 | 결과 |
| --- | --- | --- |
| 전체 IPC가 `G16H-010/60,[G06Q-010/10, G16H-010/20, G16H-080/00]`와 정확히 일치하는 특허 | Self-Query Retriever 단독 | 전체 IPC가 질문과 정확히 일치하는 특허(병원 고객관리 시스템) 반환 |
| "기능성위장관질환의 증상 조절을 위한 자가완성형 음식 조절 서비스"와 관련한 특허 | 최종 체인 → search | 식이 · 건강관리 분야 특허 4건 반환, 모두 `G16H-020/60` 계열 |
| 출원인이 특정 법인인 특허 3개 | 최종 체인 → search | 출원인 조건에 맞는 특허를 양식대로 반환 |
| 음식 조절 가이드라인 발명 설명의 IPC 생성 | 최종 체인 → generator | `G16H 50/20` · `G06Q 50/22` 등을 부여 사유와 함께 생성 (괄호 설명은 LLM 출력이며 실제 IPC 정의와의 일치는 검증하지 않음) |

**수상** 강남대학교 데이터사이언스전공 DS학술제 모델링 경진대회 장려상 (2024.11)

**남은 한계**

- 자동화 테스트는 없습니다. 모든 확인은 Colab에서 질의를 넣고 결과를 직접 보는 방식이었습니다.
- Retriever 비교는 반환 문서를 직접 확인한 정성 비교이고 정량 지표는 쓰지 않았습니다. 이 점은 노트북에도 적어 두었습니다.
- IPC 생성 결과의 정확도는 측정하지 않았습니다.
- 라우터의 분류는 LLM이 하므로 잘못 분류할 가능성이 남아 있습니다. 검색 결과가 비었을 때 생성으로 넘어가는 search 프롬프트 문장도 지우지 못했습니다.

## 실행 방법

1. Google Colab에서 `RAG_Modeling.ipynb`를 열고, Drive 작업 경로(`/content/drive/MyDrive/Colab Notebooks/testdata/contest/`)에 대회 데이터(Train · Valid 엑셀)를 넣습니다.
2. 노트북의 `pip install -U` 셀 대신 `pip install -r requirements.txt`를 실행합니다. LangChain 1.0에서 제거된 API를 쓰므로 0.3.x로 고정돼 있습니다.
3. Colab 보안 비밀에 `OPENAI_API_KEY`를 등록합니다. Colab 밖에서는 `.env.example`을 `.env`로 복사해 채웁니다.
4. Ⅰ(데이터 정제)과 Ⅲ(벡터 환경 구성)을 실행합니다. 최종 체인이 쓰는 방법 3은 반드시 실행한 뒤 `question("질문")`으로 질의합니다. 저장된 벡터DB와 `list.pickle`이 있으면 생성 셀을 건너뛰고 불러옵니다.

## 프로젝트 구조 · 보안

```
Patent_Classification/
├── RAG_Modeling.ipynb      # 전처리 → 벡터DB → Retriever → Agent/Chain, 출력 제거본
├── requirements.txt        # 의존성 목록 (LangChain 0.3.x 고정)
├── .env.example            # 환경변수 템플릿
├── .githooks/pre-commit    # 커밋 전 API 키 유출 검사
└── README.md
```

- `.env`, 대회 데이터(`data/`, `*.xlsx`), 벡터DB(`store/`), 요약 캐시(`*.pickle`)는 `.gitignore`로 제외됩니다.
- 클론 후 `git config core.hooksPath .githooks`를 실행하면 커밋할 때 API 키 패턴과 `.env` 파일을 검사합니다.
