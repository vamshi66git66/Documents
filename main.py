from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

def main():
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    # Create or load vector store
    vector_store = VectorStore()
    db = vector_store.create_or_load(documents)
    
    # Create RAG chain
    rag = RAGChain(db)
    
    # Interactive query loop
    print("RAG System Ready! Type 'quit' to exit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'quit':
            break
            
        try:
            answer = rag.query(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 