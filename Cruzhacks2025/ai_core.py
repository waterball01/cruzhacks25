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
    for page in pages:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(["Extract all the text from this image, do not include any other words oustide the text:", page])
        words.append(response.text)
    return "\n".join(words)

def extract_text_from_file(filepath):
    if filepath.endswith('.txt'):
        with open(filepath, 'r') as f:
            return f.read()
    elif filepath.endswith('.pdf'):
        return extract_from_pdf(filepath)
    elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(filepath)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(["Extract all the text from this image, do not include any other words oustide the text:", image])
        return response.text
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
    return chunks

def gemini_chat(session_id, question, db_session):
    history = load_chat_history(db_session, session_id)
    model = genai.GenerativeModel('gemini-2.0-flash-lite', system_instruction="You are a helpful expert tutor. Your users are asking questions about information provided by their lecture notes or other sources regarding their class. You will be shown the user's question, and the relevant information, answer the user's question using only this information.")
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

def generate_study_questions(filepaths, numqs, specification, session_id):
    query = "Generate study questions based on the following lecture notes. Only include information from the lecture notes, do not include any other information. Make sure to hit on all major points. Make sure to have" + str(numqs) + " questions. Make sure to include the following specifications: " + str(specification) + ". Do not include any oother characters aside from the question itself, do not include the question numbers. Add a new lince charcter between each question.\n\n"
    db_session = init_db()
    for filepath in filepaths:
        chroma(filepath,chroma_collection)
    retrieved = retrieve(query, chroma_collection)
    combined_info = "\n".join(retrieved)
    response = gemini_chat(session_id, f"{query}\n\nContext:\n{combined_info}", db_session)
    data = response.split('\n\n')
    db_session.close()
    return data

def evaluate_answers(questions, answers, session_id):
    query_prefix = (
        "Correct the following questions and answers. For each one, include:\n"
        "- Feedback on the given answer\n"
        "- The correct answer\n"
        "- A brief explanation of how to get the correct answer\n"
        "- Do not include the question itself simply follow the following format for each question\n"
        "Feedback: \n"
        "Correct Answer: \n"
        "Explanation: \n"
        "Include a newline in between responses to different questions\n\n"
    )
    db_session = init_db()
    combined_prompt_parts = []
    combined_context = []
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        retrieved = retrieve(q + a, chroma_collection)
        combined_prompt_parts.append(f"Q{i+1}: {q}\nA{i+1}: {a}")
        combined_context.extend(retrieved)

    full_prompt = (
        query_prefix +
        "\n\n".join(combined_prompt_parts) +
        "\n\nContext:\n" +
        "\n\n".join(set(combined_context))
    )

    response = gemini_chat(session_id, full_prompt, db_session)
    data = response.split('\n\n')
    db_session.close()
    return data

def answer_question(filepaths, question, session_id):
    query = "Answer the following question based on the following lecture notes and your history talking with the user. Only include information from the lecture notes, do not include any other information. Question: " + question + "\n\n"
    db_session = init_db()
    for filepath in filepaths:
        chroma(filepath,chroma_collection)
    retrieved = retrieve(question, chroma_collection)
    combined_info = "\n".join(retrieved)
    response = gemini_chat(session_id, f"{query}\n\nContext:\n{combined_info}", db_session)
    db_session.close()
    return response

FILEPATHS = ["/Users/Ruhan/Desktop/cruzhacks25/cruzhacks25/Cruzhacks2025/input/Math 145 - Sp 25 - Lecture 1.pdf","/Users/Ruhan/Desktop/cruzhacks25/cruzhacks25/Cruzhacks2025/input/Lecture 2.pdf"]
SESSION_ID = "session1234"
chroma_collection = chroma_client.create_collection(SESSION_ID, embedding_function=embedding_function)

questions = generate_study_questions(FILEPATHS, 4, "I have trouble with dynamical systems so please include them", SESSION_ID)
print(questions)
answers = ["A discrete dynamical system is a system that evolves over time in discrete steps. It is a mathematical model that describes how a system changes from one state to another. It relates to chaos because chaotic systems are often modeled as discrete dynamical systems, where small changes in initial conditions can lead to vastly different outcomes.", "I'm not sure", "idk", "I have no idea"]
response = evaluate_answers(questions, answers, SESSION_ID)
print(response)
