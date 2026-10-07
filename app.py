import re
from flask import Flask, render_template, request, jsonify
from utils.skill_matching import extract_job_skills, extract_resume_skills, match_skills
import joblib
import pdfplumber
from utils.ats_score import calculate_ats_score
from utils.suggestions import generate_suggestions, generate_suggestions_llm

app = Flask(__name__)

# ---- Load your saved models directly here ----
rf_model = joblib.load("models/resume_analyzer_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

SKILL_LIST = [
    "python", "react", "node.js", "node", "docker", "aws", "azure", "gcp",
    "machine learning", "ml", "sql", "java", "flask", "django", "javascript",
    "typescript", "kubernetes", "ci/cd", "rest api", "api", "nlp", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "git", "github", "linux",
]


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_skills(text):
    text_lower = text.lower()
    matched_skills = [
        skill for skill in SKILL_LIST
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    ]
    print(f"extract_skills matched: {matched_skills}")
    return matched_skills


def get_match_score(resume_text, jd_text, matching_skills, jd_skills):
    combined_text = resume_text + " " + jd_text
    text_vector = tfidf_vectorizer.transform([combined_text])
    predicted_score = rf_model.predict(text_vector)[0]
    return round(float(predicted_score), 2)

@app.route('/')
def home():
    return render_template('index.html')

# after button press code
@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files.get('resume')
    jd_text = request.form.get('jd_text', '')

    if not resume_file or not jd_text:
        return jsonify({"error": "Resume file and JD text are required"}), 400


    resume_text = extract_text_from_pdf(resume_file)
    resume_skills = extract_resume_skills(resume_text)
    jd_skills = extract_job_skills(jd_text)
    # Determine if JD skill extraction succeeded
    jd_extraction_failed = not bool(jd_skills)
    if jd_extraction_failed:
        print("JD SKILL EXTRACTION FAILED")

    matching_skills, missing_skills = match_skills(resume_skills, jd_skills)
    match_score = get_match_score(resume_text, jd_text, matching_skills, jd_skills)
    ats_result = calculate_ats_score(resume_text, "pdf", resume_skills)
    # If JD extraction failed, don't claim there are no missing skills — return a helpful message.
    if jd_extraction_failed:
        suggestions = [
            "Unable to identify required skills from this job description. Try including a 'Required skills' section or rephrase the JD."
        ]
    else:
        suggestions = generate_suggestions(match_score, missing_skills, ats_result["feedback"])

    final_response = {
        "match_score": match_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ats_score": ats_result["ats_score"],
        "ats_feedback": ats_result["feedback"],
        "suggestions": suggestions,
        "jd_extraction_failed": jd_extraction_failed,
    }
    print("========== BACKEND FINAL RESPONSE ==========")
    print(final_response)

    return jsonify(final_response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

