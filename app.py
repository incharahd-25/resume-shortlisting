from flask import Flask, render_template, request
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():

    job_description = request.form["jobdesc"]
    files = request.files.getlist("resume")

    results = []

    for file in files:

        text = extract_text(file)

        documents = [text, job_description]

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(matrix[0:1], matrix[1:2])

        score = round(similarity[0][0] * 100, 1)

        status = "Shortlisted" if score > 20 else "Rejected"

        results.append({
            "name": file.filename,
            "score": score,
            "status": status
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return render_template("result.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
