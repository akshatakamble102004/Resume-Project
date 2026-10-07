from io import BytesIO

import app as app_module
from utils.skill_matching import extract_job_skills, extract_resume_skills, match_skills


def test_analyze_extracts_hardware_support_skills_from_job_description(monkeypatch):
    jd_text = (
        "Hardware Support / Hardware Support Engineer role requiring skills such as: "
        "computer hardware, hardware troubleshooting, technical support, Windows, Linux, "
        "networking, LAN/WAN, troubleshooting"
    )
    resume_text = "Skills: Python, Machine Learning, Java, React, SQL"

    monkeypatch.setattr(app_module, "extract_text_from_pdf", lambda file: resume_text)
    monkeypatch.setattr(app_module, "get_match_score", lambda *args, **kwargs: 0.42)
    monkeypatch.setattr(
        app_module,
        "calculate_ats_score",
        lambda resume_text, file_type, resume_skills: {"ats_score": 75, "feedback": []},
    )
    monkeypatch.setattr(app_module, "generate_suggestions", lambda *args, **kwargs: ["stub"])

    client = app_module.app.test_client()
    response = client.post(
        "/analyze",
        data={
            "resume": (BytesIO(b"%PDF-1.4"), "resume.pdf"),
            "jd_text": jd_text,
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert any(skill.lower() == "hardware troubleshooting" for skill in payload["missing_skills"])
    assert any(skill.lower() == "networking" for skill in payload["missing_skills"])
    assert any(skill.lower() == "technical support" for skill in payload["missing_skills"])


def test_extract_job_skills_handles_hardware_support_formats():
    jd_text = (
        "Hardware Support Engineer\n\nResponsibilities:\n"
        "Provide technical support to users.\n"
        "Troubleshoot computer hardware and networking issues.\n"
        "Support Windows and Linux systems.\n"
        "Perform LAN/WAN troubleshooting.\n\n"
        "Required skills:\n"
        "Computer hardware\n"
        "Hardware troubleshooting\n"
        "Technical support\n"
        "Windows\n"
        "Linux\n"
        "Networking\n"
        "LAN/WAN\n"
        "Troubleshooting\n"
    )

    skills = extract_job_skills(jd_text)
    expected = {
        "Computer hardware",
        "Hardware troubleshooting",
        "Technical support",
        "Windows",
        "Linux",
        "Networking",
        "LAN/WAN",
        "Troubleshooting",
    }

    assert set(skills) == expected


def test_extract_job_skills_avoids_invented_hardware_skills():
    jd_text = (
        "Develop, deploy, and maintain applications on cloud platforms.\n"
        "Work with cloud services such as AWS, Azure, or Google Cloud Platform (GCP).\n"
        "Design and manage scalable and reliable cloud-based applications.\n"
        "Deploy applications using services such as EC2, S3, Lambda, RDS, and VPC.\n"
        "Work with Docker and CI/CD pipelines for application deployment.\n"
        "Monitor application performance, availability, and cloud resources.\n"
        "Implement basic cloud security, authentication, and access control.\n"
        "Work with databases such as MySQL, PostgreSQL, or MongoDB in cloud environments.\n"
        "Troubleshoot deployment, networking, and infrastructure-related issues.\n"
        "Collaborate with developers and DevOps teams to improve application deployment and reliability."
    )

    skills = extract_job_skills(jd_text)

    assert "Hardware troubleshooting" not in skills
    assert "AWS" in skills or "aws" in skills
    assert "Azure" in skills or "azure" in skills
    assert "GCP" in skills or "gcp" in skills
    assert "Docker" in skills or "docker" in skills
    assert "CI/CD" in skills or "ci/cd" in skills
    assert "Networking" in skills or "networking" in skills


def test_extract_resume_skills_from_experience_without_skills_heading():
    resume_text = (
        "Software Engineer\n"
        "Developed applications using Java and Spring Boot with MySQL.\n"
        "Built REST APIs in Python and Django for internal tools."
    )

    skills = extract_resume_skills(resume_text)

    assert any(skill.lower() == "java" for skill in skills)
    assert any(skill.lower() == "spring boot" for skill in skills)
    assert any(skill.lower() == "mysql" for skill in skills)
    assert any(skill.lower() == "python" for skill in skills)


def test_match_skills_does_not_overmatch_unrelated_technologies():
    resume_skills = ["Java", "MySQL", "Spring Boot"]
    jd_skills = ["JavaScript", "PostgreSQL", "Spring Boot"]

    matching, missing = match_skills(resume_skills, jd_skills)

    assert matching == ["Spring Boot"]
    assert missing == ["JavaScript", "PostgreSQL"]
