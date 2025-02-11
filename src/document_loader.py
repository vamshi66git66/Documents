from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

class DocumentLoader:
    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), documents_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            print(f"Created documents directory at: {self.documents_dir}")
            print("Please add your PDF documents to this directory and run again.")
            return []

        documents = []
        for file in os.listdir(self.documents_dir):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.documents_dir, file)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        if not documents:
            print(f"No PDF documents found in {self.documents_dir}")
            return []
            
        return self.text_splitter.split_documents(documents) 