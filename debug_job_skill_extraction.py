from utils.skill_matching import extract_job_skills, match_skills
import app as app_module

resume_text = """Software / Machine Learning Developer

Skills:
Python, Machine Learning, Java, React, JavaScript, SQL, Node.js

Experience:
Developed machine learning models and web applications.
"""

jd_text = """Hardware Support Engineer

Responsibilities:
Provide technical support to users.
Troubleshoot computer hardware and networking issues.
Support Windows and Linux systems.
Perform LAN/WAN troubleshooting.

Required skills:
Computer hardware
Hardware troubleshooting
Technical support
Windows
Linux
Networking
LAN/WAN
Troubleshooting
"""

print('=== RESUME TEXT ===')
print(resume_text)
print('=== JD TEXT ===')
print(jd_text)

resume_skills = app_module.extract_skills(resume_text)
print('=== EXTRACTED RESUME SKILLS ===')
print(resume_skills)

required_skills = extract_job_skills(jd_text)
print('=== EXTRACTED JD SKILLS ===')
print(required_skills)

matching_skills, missing_skills = match_skills(resume_skills, required_skills)
print('=== MATCHING SKILLS ===')
print(matching_skills)
print('=== MISSING SKILLS ===')
print(missing_skills)
