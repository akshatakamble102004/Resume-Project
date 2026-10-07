# Resume & Job Description ATS Analyzer

An AI-powered ATS (Applicant Tracking System) resume analyzer built with Flask, Machine Learning (Random Forest & TF-IDF), and OpenAI. It parses resumes in PDF format, extracts key skills, computes match scores against a given Job Description (JD), and provides ATS compatibility feedback along with actionable improvement suggestions.

## 🚀 Features

- **Resume Parsing**: Extracts structured text from PDF resumes.
- **Skill Extraction & Matching**: Identifies core technical competencies and compares candidate skills against the JD.
- **ML Match Score**: Evaluates resume-job description fit using a trained TF-IDF vectorizer and Random Forest model.
- **ATS Compatibility Scoring**: Analyzes section headers, formatting, word count, and keyword density.
- **Actionable Feedback**: Generates tailored suggestions to improve job match and pass ATS screening.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **Machine Learning / NLP**: Scikit-learn, Sentence-Transformers, Joblib, SpaCy
- **Document Processing**: pdfplumber
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Gunicorn, Render / Railway

---

## 📂 Project Structure

```text
├── models/
│   ├── resume_analyzer_model.pkl    # Pre-trained ML model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
├── static/
│   └── style.css                    # UI styles
├── templates/
│   └── index.html                   # Web interface
├── utils/
│   ├── ats_score.py                 # ATS scoring logic
│   ├── skill_matching.py            # Skill extraction & matching algorithms
│   └── suggestions.py               # Recommendation generators
├── app.py                           # Flask application entry point
├── Procfile                         # Cloud process configuration
├── requirements.txt                 # Dependencies
└── .env.example                     # Environment variables template
```

---

## ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshatakamble102004/Resume-Project.git
   cd Resume-Project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Copy `.env.example` to `.env` and configure your API key (if using OpenAI suggestions):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌐 Deployment to Render

1. Create a free account at [Render](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository: `akshatakamble102004/Resume-Project`.
4. Configure the service settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. In **Environment Variables**, add:
   - `OPENAI_API_KEY` (if utilizing OpenAI-powered suggestions)
6. Click **Deploy Web Service**.
