from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config import LLM_MODEL



def reranker (docs:list[Document], query:str):
    rerank_prompt = PromptTemplate.from_template(''' You are a helpful reranker assistant i.e. 
    given a question & a list of documents your task is to rank them in order considering the given question.

    Question:{question}
                                                    
    Documents:{docs}
                                                    
    Instructions:
    - Given the question, find out the relevance of each document.
    - Only return document indices from the order of most relevant to least relevant, no explanation needed
    ''')
    doc_lines = [f'{i+1}. {doc.page_content}' for i,doc in enumerate(docs)]
    docs_text = '\n\n'.join(doc_lines)

    reranker_llm=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')
    reranker_chain = rerank_prompt | reranker_llm | StrOutputParser()
    rank_res = reranker_chain.invoke({'question':query,'docs':docs_text})

    try:
        indices = [int(i.strip()) - 1 for i in rank_res.split(",")]
        # ✅ Index into original docs list, not the string
        reranked_docs = [docs[i] for i in indices if 0 <= i < len(docs)]
    except (ValueError, IndexError):
        print("Reranker output could not be parsed, returning original order.")
        reranked_docs = docs
    return reranked_docs