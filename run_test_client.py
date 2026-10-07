from io import BytesIO
import app as app_module

# Monkeypatch PDF extraction to avoid using a real PDF file
app_module.extract_text_from_pdf = lambda f: 'Experience\nSkills\nPython, Machine Learning, Java, React, SQL\n'

jd_text = 'Hardware Support / Hardware Support Engineer role requiring skills such as: computer hardware, hardware troubleshooting, technical support, Windows, Linux, networking, LAN/WAN, troubleshooting'

client = app_module.app.test_client()
resp = client.post('/analyze', data={'resume': (BytesIO(b'%PDF-1.4'), 'resume.pdf'), 'jd_text': jd_text}, content_type='multipart/form-data')
print(resp.get_json())
