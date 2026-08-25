from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from reranker import reranker
from config import LLM_MODEL

_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

def clear_session(session_id: str = "default") -> None:
    if session_id in _store:
        _store[session_id].clear()

def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        url = doc.metadata.get("yt_url", "no timestamp")
        parts.append(f"[Source: {url}]\n{doc.page_content}")
    return "\n\n".join(parts)

def retrieve_and_rerank(retriever, query: str) -> str:
    retrieved = retriever.invoke(query)
    reranked = reranker(docs=retrieved, query=query)
    return format_docs(reranked)

def build_chain(retriever):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about a YouTube video.
    Use ONLY the transcript context below to answer. 
    When referencing a specific moment, include the timestamp URL so the user can jump to it.
    If the context does not contain enough information, say you don't know — do not make things up.

    Context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
    ])

    core_chain = (
        RunnablePassthrough.assign(
            # Retrieve + rerank on every question turn using the current question
            context=RunnableLambda(
                lambda x: retrieve_and_rerank(retriever, x["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )