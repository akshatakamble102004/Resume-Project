import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ats_score import calculate_ats_score


def test_perfect_resume_score_is_capped_at_100():
    resume_text = """
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
    """
    result = calculate_ats_score(resume_text, "pdf", ["python", "react", "docker", "aws", "sql"])
    assert 0 <= result["ats_score"] <= 100


def test_weak_resume_score_is_between_zero_and_hundred():
    result = calculate_ats_score("No useful content here", "txt", [])
    assert 0 <= result["ats_score"] <= 100


def test_backend_score_is_not_scaled_twice():
    result = calculate_ats_score("Experience Education Skills Summary Email: a@b.com Phone: 1234567890", "pdf", ["python"])
    assert isinstance(result["ats_score"], (int, float))
    assert 0 <= result["ats_score"] <= 100
