import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import declarative_base

# --- Gemini API setup ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# --- ChromaDB setup ---
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name="output",
    embedding_function=embedding_function
)

# --- Cross-encoder setup ---
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- SQLAlchemy base setup ---
Base = declarative_base()

# --- Export for import in other modules ---
__all__ = [
    "genai",
    "chroma_collection",
    "cross_encoder",
    "Base"
]