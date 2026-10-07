import re


def _normalize_skills(resume_skills):
    if resume_skills is None:
        return []

    if isinstance(resume_skills, str):
        return [
            skill.strip()
            for skill in resume_skills.split(",")
            if skill.strip()
        ]

    if isinstance(resume_skills, (list, tuple, set)):
        return [
            str(skill).strip()
            for skill in resume_skills
            if str(skill).strip()
        ]

    return []


def calculate_ats_score(resume_text, file_type, resume_skills, required_skills=None):
    """
    Rule-based ATS score calculator.

    Score = 100 points total

    1. File Type              -> 10 points
    2. Resume Sections        -> 20 points
    3. Bullet Points          -> 15 points
    4. Relevant Skills        -> 35 points
    5. Resume Length          -> 10 points
    6. Contact Information    -> 10 points

    Returns:
        {
            "ats_score": score,
            "feedback": [...]
        }
    """

    if resume_text is None:
        resume_text = ""

    resume_text = str(resume_text)
    resume_text_lower = resume_text.lower()

    valid_resume_skills = _normalize_skills(resume_skills)
    valid_required_skills = _normalize_skills(required_skills)

    score = 0
    feedback = []

    # 1. FILE TYPE - 10 POINTS
    file_type_value = str(file_type or "").lower().replace(".", "").strip()
    if file_type_value in ["pdf", "docx"]:
        score += 10
    else:
        feedback.append("Use PDF or DOCX format for better ATS compatibility.")

    # 2. RESUME SECTIONS - 20 POINTS
    section_groups = {
        "experience": ["experience", "work experience", "professional experience", "employment"],
        "education": ["education", "academic background", "academic qualification"],
        "skills": ["skills", "technical skills", "core skills"],
        "summary": ["summary", "professional summary", "career objective", "objective", "profile"],
    }

    found_sections = []
    for section, keywords in section_groups.items():
        if any(keyword in resume_text_lower for keyword in keywords):
            found_sections.append(section)

    section_score = (len(found_sections) / len(section_groups)) * 20
    score += section_score

    if len(found_sections) < len(section_groups):
        missing_sections = sorted(set(section_groups.keys()) - set(found_sections))
        feedback.append("Add missing sections: " + ", ".join(missing_sections))

    # 3. BULLET POINTS - 15 POINTS
    bullet_count = resume_text.count("•") + len(re.findall(r"(?m)^\s*[-*]\s+", resume_text))
    if bullet_count >= 5:
        score += 15
    else:
        bullet_score = (bullet_count / 5) * 15
        score += bullet_score
        feedback.append("Use more bullet points to clearly describe your experience and responsibilities.")

    # 4. RELEVANT SKILLS - 35 POINTS
    if valid_required_skills:
        matching_required_skills = [
            skill for skill in valid_required_skills
            if skill.lower() in {resume_skill.lower() for resume_skill in valid_resume_skills}
        ]
        if valid_required_skills:
            relevant_skill_percentage = len(matching_required_skills) / len(valid_required_skills)
            skill_score = relevant_skill_percentage * 35
            score += skill_score
        else:
            skill_score = 0
            score += skill_score

        if skill_score < 35:
            feedback.append("Add more relevant skills from the job description.")
    else:
        if valid_resume_skills:
            skill_score = min(len(valid_resume_skills) / 8, 1.0) * 35
            score += skill_score
            if skill_score < 35:
                feedback.append("Add more relevant technical skills and keywords from the job description.")
        else:
            score += 0
            feedback.append("Add more relevant technical skills and keywords from the job description.")

    # 5. RESUME LENGTH - 10 POINTS
    words = resume_text.split()
    word_count = len(words)
    if 300 <= word_count <= 1000:
        score += 10
    elif word_count < 300:
        length_score = (word_count / 300) * 10
        score += min(length_score, 10)
        feedback.append("Resume seems too short. Add more details about projects, experience, and responsibilities.")
    else:
        length_score = max(0, 10 - ((word_count - 1000) / 500) * 10)
        score += length_score
        feedback.append("Resume seems too long. Try to keep it concise and relevant.")

    # 6. CONTACT INFORMATION - 10 POINTS
    contact_score = 0
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    if re.search(email_pattern, resume_text):
        contact_score += 5
    else:
        feedback.append("Add a professional email address.")

    phone_pattern = r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"
    if re.search(phone_pattern, resume_text):
        contact_score += 5
    else:
        feedback.append("Add a valid phone number.")

    score += contact_score

    score = max(0, min(100, round(score, 2)))

    return {"ats_score": score, "feedback": feedback}