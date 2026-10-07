from io import BytesIO
import app as app_module

app_module.extract_text_from_pdf = lambda file: 'Skills: Python, Machine Learning, Java, React, SQL'
app_module.get_match_score = lambda *args, **kwargs: 0.73
app_module.calculate_ats_score = lambda resume_text, file_type, resume_skills: {'ats_score': 75.0, 'feedback': []}
app_module.generate_suggestions = lambda *args, **kwargs: ['stub']

client = app_module.app.test_client()
response = client.post(
    '/analyze',
    data={
        'resume': (BytesIO(b'%PDF-1.4'), 'resume.pdf'),
        'jd_text': 'Hardware Support / Hardware Support Engineer role requiring skills such as: computer hardware, hardware troubleshooting, technical support, Windows, Linux, networking, LAN/WAN, troubleshooting'
    },
    content_type='multipart/form-data'
)
print(response.get_json())
