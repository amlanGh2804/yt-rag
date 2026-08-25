from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def get_vectorstore() -> Chroma:
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=PERSIST_DIR,
    )

def index_docs(vectorstore: Chroma, docs: list[Document], video_id: str):
    existing = vectorstore.get(where={"video_id": video_id})
    if len(existing['ids']) == 0:
        vectorstore.add_documents(docs)
        print(f'Indexed {len(docs)} chunks.')
    else:
        print(f'Already indexed {len(existing['ids'])} chunks, skipping.')