from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from ai_core import generate_study_questions, evaluate_answers, answer_question
from flask import jsonify
import os
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from config import genai, chroma_collection, cross_encoder, Base

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")
        file_paths = []

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)

        question_count = int(request.form["question_count"])
        prompt = request.form["prompt"]

        session["file_paths"] = file_paths
        session["question_count"] = question_count
        session["prompt"] = prompt

        print("FILES:", file_paths)
        questions = generate_study_questions(file_paths, prompt, question_count, session_id=90210)
        session["questions"] = questions

        return redirect(url_for("quiz"))

    return render_template("index.html")


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    questions = session.get("questions", [])
    if request.method == "POST":
        answers = [request.form.get(f"answer_{i}") for i in range(len(questions))]
        session["answers"] = answers
        return redirect(url_for("results"))

    return render_template("quiz.html", questions=questions)


@app.route("/results")
def results():
    questions = session.get("questions", [])
    answers = session.get("answers", [])
    feedback = evaluate_answers(questions, answers,session_id=90210)
    qa_pairs = zip(questions, answers, feedback)
    return render_template("results.html", qa_pairs=qa_pairs)


@app.route("/clarify", methods=["POST"])
def clarify():
    data = request.get_json()
    question = data.get("question", "")
    answer = answer_question(question, session_id=90210)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)