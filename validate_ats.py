from utils.ats_score import calculate_ats_score

cases = [
    (
        "Perfect resume",
        """
        Experience
        Education
        Skills
        Summary
        John Doe
        Email: john@example.com
        Phone: +1 555-123-4567
        - Built scalable web applications.
        - Led a cross-functional team.
        - Delivered APIs and dashboards.
        - Improved reliability and performance.
        - Mentored engineers and owned releases.
        Python React Node.js Docker AWS SQL JavaScript
        """,
        "pdf",
        ["python", "react", "docker", "aws", "sql"],
        None,
    ),
    (
        "Weak resume",
        "No useful content here",
        "txt",
        [],
        None,
    ),
    (
        "Backend score case",
        "Experience Education Skills Summary Email: a@b.com Phone: 1234567890",
        "pdf",
        ["python"],
        ["python", "react"],
    ),
]

for name, resume, file_type, resume_skills, required_skills in cases:
    result = calculate_ats_score(resume, file_type, resume_skills, required_skills)
    print(name, '=>', result['ats_score'], 'feedback=', result['feedback'])
    assert 0 <= result['ats_score'] <= 100
