# rag_engine.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

DB_DIR = "./chroma_db"

# 1. Initialize Open Source Embeddings Engine
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_vector_db():
    """Seeds a local ChromaDB with verified fact-checked reference data."""
    # Mock trusted facts database
    trusted_facts = [
        "NASA officially confirmed that no alien life forms or extraterrestrial spaceships have landed on Earth.",
        "The economic bill signed this month focuses on infrastructure tax credits and local manufacturing incentives.",
        "Medical health associations state that chocolate contains antioxidants but cannot cure chronic illnesses or diseases.",
        "The federal election commission verified that local election results across all districts were audited and validated safely."
    ]
    
    docs = [Document(page_content=text, metadata={"source": f"fact_{i}"}) for i, text in enumerate(trusted_facts)]
    
    # Create and persist vector store
    print("Vectorizing verified repositories into ChromaDB...")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
    print("Vector store initialized successfully.")
    return vector_store

def get_vector_db():
    """Loads the existing vector database instance."""
    if not os.path.exists(DB_DIR):
        return initialize_vector_db()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def query_trusted_facts(query: str, k: int = 2):
    """Queries the database for matching trusted articles."""
    db = get_vector_db()
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    # Run standalone to initialize the database
    initialize_vector_db()