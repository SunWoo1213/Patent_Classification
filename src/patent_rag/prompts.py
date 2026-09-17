"""프롬프트 및 메타데이터 필드 정의."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# ---------------------------------------------------------------------------
# 방법1: 문서 요약
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in summarizing documents in Korean."),
    ("user", "Summarize the following documents in 3 sentences in bullet points format.\n\n{doc}"),
])

# ---------------------------------------------------------------------------
# 방법3: Self-Query Retriever 메타데이터 설명
# ---------------------------------------------------------------------------
DOCUMENT_CONTENTS = "특허 정보를 가지고 있다"

METADATA_FIELDS = [
    ("doc_id", "doc_id 는 이 문서를 다른 문서와 연결해 주는 key 값이다"),
    ("메인IPC2", "특허에서 주요 주제에 해당하는 IPC 코드이다."),
    ("전체 IPC", "특허의 요약내용을 토대로 이 특허에 해당하는 IPC 전체 값을 가지고 있다. "
                "사용자 요청이 IPC를 포함하고 있을 때는 반드시 '전체 IPC'를 이용하여 정확히 일치하는 "
                "데이터를 먼저 검색하고 그 다음에 메인IPC2를 검색한다."),
    ("출원번호", "특허의 출원번호를 저장하고 있다."),
    ("등록번호", "특허의 등록번호를 저장하고 있다."),
    ("출원인", "특허를 출원한 회사 또는 개인 정보를 저장하고 있다."),
    ("발명자", "특허를 발명한 자의 정보를 보관하고 있다."),
    ("법적상태", "특허를 현재 상태 정보를 보관하고 있다."),
]

# ---------------------------------------------------------------------------
# Retriever Agent
# ---------------------------------------------------------------------------
TOOL_DESCRIPTION = (
    "사용자의 특허 관련 질문을 검색합니다. 만약 여기에 존재하지 않는 특허와 관련한 질문이라면 LLM을 통해 "
    "검색합니다. 만약 사용자의 질문에 IPC가 포함되어 있다면 반드시 여기에서 얻은 답변에도 동일한 IPC가 "
    "포함되어야 합니다."
)

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.\n"
               "만약 사용자의 질문에 IPC 정보가 포함되어 있다면 IPC 문자열을 이용해서 정보를 검색해 주세요."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ---------------------------------------------------------------------------
# Router Chain
# ---------------------------------------------------------------------------
SEARCH_PROMPT = PromptTemplate.from_template(
    """당신은 특허 전문가 입니다.
당신은 아래에 '제공되는 정보'를 기준으로 '사용자 요청'과 일치하는 정보를 제공합니다.
아울러 아래의 '제공되는 정보' 에서 '출원번호', '메인IPC', '전체IPC', '요약정보', '법적상태', '출원인', '등록번호' 정보를 추출합니다.
만약 정보를 '제공되는 정보'에서 추출하지 못할 경우 '검색불가' 로 기재합니다.
아울러 '제공되는 정보' 가 없을 경우에는 당신이 생성합니다.
만약 추출한 '요약정보'가 100자 이상일 경우 이를 50자 이내로 요약합니다.

'사용자 요청' : {input}
'제공되는 정보' : {context}

출력시에 아래의 규칙에 따라 출력해 줍니다.
'출원번호' :
'메인IPC'  :
'전체IPC'  :
'요약정보' :
'법적상태' :
'등록번호' :
'출원인'   :"""
)

GENERATOR_PROMPT = PromptTemplate.from_template(
    """당신은 특허코드인 IPC 를 생성해 주는 전문가 입니다.
당신은 사용자가 입력한 {input} 을 가지고 국제 기준에 맞는 특허 IPC 코드를 생성합니다.
아래와 같은 작업을 단계별로 수행합니다.
1. 가장 중요한 내용 순으로 정보를 추출합니다.
2. 각 추출된 내용을 가지고 IPC 코드, 한글코드, 사유를 생성합니다. 최대 10개까지 생성합니다.
3. 아래의 출력 형식을 사용하여 출력합니다.
[출력형식]
1. 만약 문장이 "다." 로 끝나면 한줄 띄어쓰기를 합니다.
2. 추출된 정보를 중요도 순서대로 기재합니다.
3. 요약된 내용은 "요약내용 : " 형태로 출력합니다.
4. "IPC :   한글코드 :    =====> 코드 부여 이유" 형태로 출력하고, 항목마다 줄을 바꿉니다."""
)

DEFAULT_PROMPT = PromptTemplate.from_template("{input}")

ROUTES = {
    "search": "기존 정보를 조회합니다",
    "generator": "새로운 정보를 생성합니다.",
}

ROUTER_PROMPT = PromptTemplate.from_template(
    """
For the given question, refer to the explanations for the following categories below and select the appropriate category.
If there is no suitable category, return the string "default".
The output should be single word

Categories are composed of strings in the format "Category: Description", and the categories are as follows:
{destinations}

Question:
{input}
"""
).partial(destinations="\n".join(f"{name}: {desc}" for name, desc in ROUTES.items()))

SEARCH_QUERY_SUFFIX = " 단, 가장 일치하는 정보를 검색해 주세요"
