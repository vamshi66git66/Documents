from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name) 