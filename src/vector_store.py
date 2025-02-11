from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
from .embeddings import get_embeddings

class VectorStore:
    def __init__(self, persist_directory: str = "../data/chroma"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        
    def create_or_load(self, documents: List[Document] = None):
        if documents:
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        ) 