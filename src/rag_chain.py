from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = Ollama(model="llama2")
        
    def create_chain(self):
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        Context: {context}
        Question: {input}

        Answer: """)

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        retriever = self.vector_store.as_retriever()
        return create_retrieval_chain(retriever, document_chain)
    
    def query(self, question: str):
        chain = self.create_chain()
        response = chain.invoke({"input": question})
        return response["answer"] 