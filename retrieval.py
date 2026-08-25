from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import TOP_K

def get_retriever(vectorstore: Chroma, docs: list[Document], video_id: str):
    dense = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': TOP_K,'filter': {'video_id': video_id}},
    )   
    sparse = BM25Retriever.from_documents(documents=docs, k=TOP_K) 

    return EnsembleRetriever(
        retrievers=[sparse, dense],
        weights=[0.3, 0.7]
    )
