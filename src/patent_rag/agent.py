"""Retriever 를 도구로 사용하는 대화형 Agent (세션별 대화 기록 유지)."""
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from patent_rag.models import get_chat_model
from patent_rag.prompts import AGENT_PROMPT, TOOL_DESCRIPTION

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_agent(retriever: BaseRetriever, tool_name: str = "Patent_inquiry",
                verbose: bool = False) -> Runnable:
    # 도구 이름은 영문이어야 한다. (한글 이름은 OpenAI API 에서 오류 발생)
    tool = create_retriever_tool(retriever, name=tool_name, description=TOOL_DESCRIPTION)
    agent = create_openai_tools_agent(get_chat_model(), [tool], AGENT_PROMPT)
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=verbose, handle_parsing_errors=True)
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def ask_agent(agent: Runnable, question: str, session_id: str = "default") -> str:
    result = agent.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
    return result["output"]
