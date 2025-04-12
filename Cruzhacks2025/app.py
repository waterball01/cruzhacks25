from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from ai_core import generate_study_questions  # Your AI function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call your AI function with the file path
    questions = generate_study_questions(file_path)

    return jsonify({"questions": questions})

if __name__ == "__main__":
    app.run(debug=True)
