import chromadb
import os
import fitz
import pytesseract
from PIL import Image
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
import uuid
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from pdf2image import convert_from_path
import cv2
import numpy as np

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

Base = declarative_base()
class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36))
    role = Column(String(10))
    message = Column(Text)
    created_at = Column(String(20), default=lambda: datetime.now(timezone.utc).isoformat())

def init_db():
    engine = create_engine("sqlite:///chat.db", future=True) 
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    return Session()

def save_message(db_session, session_id, role, message):
    chat_entry = ChatHistory(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        message=message
    )
    db_session.add(chat_entry)
    db_session.commit()

def load_chat_history(db_session, session_id):
    history = db_session.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at).all()
    
    return [{"role": entry.role, "parts": [entry.message]} for entry in history]

def extract_from_pdf(filepath):
    pages = convert_from_path(filepath)
    words = []
    for image in pages:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        words.append(pytesseract.image_to_string(rotated))
    return "\n".join(words)

def extract_text_from_file(filepath):
    if filepath.endswith('.txt'):
        with open(filepath, 'r') as f:
            return f.read()
    elif filepath.endswith('.pdf'):
        return extract_from_pdf(filepath)
    elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(filepath)
        return pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text, max_len=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_len:
            current += para + " "
        else:
            chunks.append(current.strip())
            current = para + " "
    if current:
        chunks.append(current.strip())
    print("Chunks:", chunks)
    return chunks

def gemini_chat(session_id, question, db_session):
    history = load_chat_history(db_session, session_id)
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction="You are a helpful expert tutor. Your users are asking questions about information provided by their lecture notes or other sources regarding their class. You will be shown the user's question, and the relevant information, answer the user's question using only this information.")
    chat = model.start_chat(history=history)
    response = model.generate_content(question,generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        temperature=0.3,
    ),)
    save_message(db_session, session_id, "user", question)
    save_message(db_session, session_id, "model", response.text)
    return response.text

def chroma(filepath, chroma_collection):
    text = extract_text_from_file(filepath)
    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    chroma_collection.add(ids=ids, documents=chunks)
    return chunks

def retrieve(query, chroma_collection, n_results=10):
    result = chroma_collection.query(query_texts=[query], n_results=n_results, include=["documents"])
    docs = result["documents"][0]
    scores = cross_encoder.predict([[query, doc] for doc in docs])
    reranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked

def ai_tutor(filepaths, query, session_id):
    db_session = init_db()
    for filepath in filepaths:
        chroma(filepath,chroma_collection)
    retrieved = retrieve(query, chroma_collection)
    combined_info = "\n".join(retrieved)
    response = gemini_chat(session_id, f"{query}\n\nContext:\n{combined_info}", db_session)
    db_session.close()


FILEPATHS = ["/Users/Ruhan/Desktop/cruzhacks25/input/Math 145 - Sp 25 - Lecture 1.pdf","/Users/Ruhan/Desktop/cruzhacks25/input/Lecture 2.pdf"]
QUERY1 = "What is a discrete dynamical system and how does it relate to chaos?"
SESSION_ID = "session1234"
chroma_collection = chroma_client.create_collection(SESSION_ID, embedding_function=embedding_function)

ai_tutor(FILEPATHS, QUERY1, SESSION_ID)