# skill_matching.py

"""Utility functions for extracting and matching skills between a resume and a job description.

The module provides:
- `extract_resume_skills` – simple heuristic to pull explicit skills from a resume string.
- `extract_job_skills` – uses the OpenAI client (if available) to obtain a JSON list of required skills.
- `normalize_skill` – applies common aliases and lower‑cases.
- `match_skills` – compares two skill lists, falling back to strict exact matching when SBERT is unavailable.

Debug prints are emitted for every stage of the pipeline using the exact labels requested by the user.
"""

import re
import json
from difflib import SequenceMatcher

# Attempt to import the OpenAI client defined elsewhere (utils.suggestions). If unavailable, we continue without it.
try:
    from utils.suggestions import client  # type: ignore
except Exception:
    client = None

# Optional SBERT embeddings – may be unavailable in the environment.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# Load SBERT model once (if the library is present).
sbert_model = None
if SentenceTransformer is not None:
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------
SKILL_ALIASES = {
    "node": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "react js": "react",
    "ml": "machine learning",
    "js": "javascript",
    "py": "python",
    "python3": "python",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "restapi": "rest api",
    "rest api": "rest api",
    "springboot": "spring boot",
    "spring boot": "spring boot",
}

GENERIC_STOPWORDS = {
    "the", "and", "or", "for", "with", "without", "using", "use", "used", "to", "of", "in", "on",
    "at", "from", "by", "as", "an", "a", "is", "are", "be", "been", "being", "developed",
    "develop", "built", "designed", "implemented", "managed", "maintained", "worked",
    "support", "supporting", "experience", "experienced", "team", "teams", "project", "projects",
    "applications", "application", "software", "developer", "engineers", "engineer", "role",
    "responsibilities", "responsibility", "tools", "tool", "services", "service", "platforms",
    "platform", "cloud", "systems", "system", "internal", "company", "business", "technical",
    "skills", "skill", "requirements", "requirement", "proficient", "familiar", "strong",
    "good", "excellent", "basic", "advanced", "knowledge", "monitor", "deploy", "deployment",
    "improve", "improving", "reliable", "scalable", "reliability", "availability", "resources",
    "performance", "issues", "related", "infrastructure", "troubleshoot", "troubleshooting",
    "security", "access", "control", "authentication", "databases", "database", "pipelines",
    "deployment", "applications", "developer", "developers"
}

GENERIC_ROLE_WORDS = {
    "software", "developer", "developers", "engineer", "engineers", "analyst", "specialist",
    "manager", "lead", "intern", "senior", "junior", "consultant", "architect", "admin"
}

JD_SKILL_PATTERNS = [
    ("Computer hardware", r"\bcomputer hardware\b|\bhardware\b"),
    ("Hardware troubleshooting", r"\bhardware troubleshooting\b|troubleshoot(?:ing)?\s+computer hardware|troubleshoot(?:ing)?\s+hardware\b"),
    ("Technical support", r"\btechnical support\b|\bhelp desk\b|\bcustomer support\b"),
    ("Windows", r"\bwindows\b"),
    ("Linux", r"\blinux\b"),
    ("LAN/WAN", r"\blan/?wan\b|local area network|wide area network"),
    ("Networking", r"\bnetwork(?:ing)?\b|\bnetworks?\b"),
    ("Python", r"\bpython\b"),
    ("Machine learning", r"\bmachine learning\b|\bml\b"),
    ("SQL", r"\bsql\b"),
    ("JavaScript", r"\bjavascript\b|\bjs\b"),
    ("React", r"\breact\b|\breact\.js\b|\breactjs\b"),
    ("node.js", r"\bnode(?:\.js)?\b|\bnodejs\b"),
    ("Docker", r"\bdocker\b"),
    ("AWS", r"\baws\b|\bamazon web services\b"),
    ("Azure", r"\bazure\b"),
    ("GCP", r"\bgcp\b|google cloud\b"),
    ("EC2", r"\bec2\b"),
    ("S3", r"\bs3\b"),
    ("Lambda", r"\blambda\b"),
    ("RDS", r"\brds\b"),
    ("VPC", r"\bvpc\b"),
    ("Cloud security", r"\bcloud security\b"),
    ("Authentication", r"\bauthentication\b"),
    ("Access control", r"\baccess control\b"),
    ("MySQL", r"\bmysql\b"),
    ("PostgreSQL", r"\bpostgresql\b"),
    ("MongoDB", r"\bmongodb\b"),
    ("Kubernetes", r"\bkubernetes\b"),
    ("CI/CD", r"\bci/cd\b|continuous integration|continuous deployment|continuous delivery\b"),
    ("REST API", r"\brest api\b|\bapi\b"),
    ("NLP", r"\bnlp\b|natural language processing\b"),
    ("Pandas", r"\bpandas\b"),
    ("NumPy", r"\bnumpy\b"),
    ("scikit-learn", r"\bscikit[- ]learn\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("PyTorch", r"\bpytorch\b"),
    ("Git", r"\bgit\b"),
    ("GitHub", r"\bgithub\b"),
]


def normalize_skill(skill: str) -> str:
    """Return a canonical lower‑cased version of *skill* applying known aliases."""
    if not skill:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", skill.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return SKILL_ALIASES.get(cleaned, cleaned)


def _dedupe_skills(skills: list) -> list:
    seen = set()
    unique = []
    for skill in skills:
        normalized = normalize_skill(skill)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(skill.strip())
    return unique


def _looks_like_skill_candidate(candidate: str) -> bool:
    if not candidate:
        return False

    cleaned = re.sub(r"[^a-z0-9]+", " ", candidate.strip().lower()).strip()
    if not cleaned:
        return False

    tokens = cleaned.split()
    if not tokens:
        return False
    if len(tokens) > 4:
        return False
    if any(token in GENERIC_STOPWORDS for token in tokens):
        return False
    if any(token in GENERIC_ROLE_WORDS for token in tokens):
        return False
    if len(tokens) == 1:
        return len(tokens[0]) >= 2 and not tokens[0].isdigit()

    if any(re.search(r"[0-9./#-]", candidate) for _ in [0]):
        return True
    if any(token.isupper() for token in candidate.split()):
        return True
    return True


def _extract_candidate_skills_from_text(text: str) -> list:
    if not text:
        return []

    candidates = []
    seen = set()
    for chunk in re.split(r"(?<=[.?!;:])\s+|\n+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        for part in re.split(r"[,;:/()]+", chunk):
            segment = part.strip(" .:-;()[]")
            if not segment:
                continue
            for subsegment in re.split(
                r"\b(?:using|with|in|on|for|via|through|such as|including|and|or)\b",
                segment,
                flags=re.IGNORECASE,
            ):
                phrase = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", subsegment).strip()
                if not phrase:
                    continue
                normalized = normalize_skill(phrase)
                if not normalized or normalized in seen:
                    continue
                if _looks_like_skill_candidate(phrase):
                    seen.add(normalized)
                    candidates.append(phrase.strip())

    return _dedupe_skills(candidates)

# ---------------------------------------------------------------------------
# Extraction helpers with required debug prints
# ---------------------------------------------------------------------------

def extract_resume_skills(resume_text: str) -> list:
    """Extract job-relevant skills from a resume using section parsing and generic text scanning."""
    print("========== RAW RESUME ==========")
    print(resume_text)

    skill_headers = r"(?:skills|technical skills|skill set|skillset|areas of expertise|areas of experience|technologies|tools|platforms|frameworks|databases|programming languages|certifications)"
    section_pattern = rf"{skill_headers}\s*:\s*([\s\S]+?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z0-9 &/]+\s*:\s*|$)"
    section_match = re.search(section_pattern, resume_text, re.IGNORECASE)

    skills = []
    if section_match:
        skills_text = section_match.group(1).strip()
        skills.extend(s for s in _split_skill_candidates(skills_text) if s)

    skills.extend(_extract_candidate_skills_from_text(resume_text))
    skills = _dedupe_skills(skills)

    print("========== RAW RESUME SKILLS ==========")
    print(skills)
    return skills


def _split_skill_candidates(skills_text: str) -> list:
    skills_text = skills_text.strip()
    if not skills_text:
        return []
    candidates = []
    for part in re.split(r"[•\n\r\-\*]+", skills_text):
        segment = part.strip()
        if not segment:
            continue
        # Support comma-separated lists, semicolons, and simple conjunctions.
        for item in re.split(r",|;|\band\b|\bor\b", segment, flags=re.IGNORECASE):
            candidate = item.strip(" .;:\t")
            if not candidate:
                continue
            if "/" in candidate and not re.search(r"\blan/?wan\b", candidate, re.IGNORECASE):
                subparts = [sub.strip() for sub in re.split(r"/|&", candidate) if sub.strip()]
                candidates.extend(subparts)
            else:
                candidates.append(candidate)
    return candidates


def _extract_skills_by_keyword(job_description: str) -> list:
    found = []
    for canonical, pattern in JD_SKILL_PATTERNS:
        if re.search(pattern, job_description, re.IGNORECASE):
            found.append(canonical)
    return _dedupe_skills(found)


def _is_explicit_skill(skill: str, job_description: str) -> bool:
    """Return True only when a skill is explicitly present in the JD text."""
    if not skill:
        return False

    skill_text = skill.strip().lower()
    if not skill_text:
        return False

    lower_jd = job_description.lower()
    if re.search(rf"(?<!\w){re.escape(skill_text)}(?!\w)", lower_jd):
        return True

    # Support a few explicit aliases/variants that are still grounded in the JD.
    alias_map = {
        "aws": [r"\baws\b", r"\bamazon web services\b"],
        "azure": [r"\bazure\b"],
        "gcp": [r"\bgcp\b", r"\bgoogle cloud(?: platform)?\b"],
        "ci/cd": [r"\bci/cd\b", r"\bcontinuous integration\b", r"\bcontinuous deployment\b", r"\bcontinuous delivery\b"],
        "machine learning": [r"\bmachine learning\b", r"\bml\b"],
        "node.js": [r"\bnode(?:\.js)?\b", r"\bnodejs\b"],
        "rest api": [r"\brest api\b", r"\bapi\b"],
        "scikit-learn": [r"\bscikit[- ]learn\b"],
    }
    for alias_pattern in alias_map.get(skill_text, []):
        if re.search(alias_pattern, lower_jd):
            return True

    return False


def _filter_explicit_skills(skills: list, job_description: str) -> list:
    explicit = []
    seen = set()
    for skill in skills:
        normalized = skill.strip().lower()
        if not normalized or normalized in seen:
            continue
        if _is_explicit_skill(skill, job_description):
            seen.add(normalized)
            explicit.append(skill.strip())
    return explicit


def _infer_skills_from_job_description(job_description: str) -> list:
    inferred = _extract_skills_by_keyword(job_description)
    if inferred:
        return inferred
    return _extract_candidate_skills_from_text(job_description)


def _extract_job_skill_section(job_description: str) -> str | None:
    section_patterns = [
        r"(?:required|essential|preferred|desired|key|core|must[- ]have|must have|primary) skills?:\s*([\s\S]+?)(?=\n\s*(?:[A-Za-z][A-Za-z ]+|[-*•])+:|\n\s*\n|$)",
        r"(?:skills and qualifications|skills/qualifications|experience with|you should have|what you will do|what you will have|you will work with)\s*:\s*([\s\S]+?)(?=\n\s*(?:[A-Za-z][A-Za-z ]+|[-*•])+:|\n\s*\n|$)",
        r"(?:responsibilities|duties|experience):\s*([\s\S]+?)(?=\n\s*(?:[A-Za-z][A-Za-z ]+|[-*•])+:|\n\s*\n|$)",
    ]
    for pattern in section_patterns:
        match = re.search(pattern, job_description, re.IGNORECASE)
        if match:
            section_text = match.group(1).strip()
            if section_text:
                return section_text
    return None


def _extract_skill_lines(job_description: str) -> list:
    lines = []
    for line in job_description.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*•]\s+.+", stripped):
            lines.append(re.sub(r"^[-*•]\s+", "", stripped))
    return lines


def _clean_skill_phrase(phrase_text: str) -> str:
    phrase_text = phrase_text.strip()
    phrase_text = re.sub(
        r"^(?:must have|should have|you should have|experience with|knowledge of|proficient in|familiar with|strong understanding of|strong knowledge of|experience in|experience supporting|including|includes|include|skilled in|skilled at|capable of|skills such as|skills include|skills include)\s*[:\-]?\s*",
        "",
        phrase_text,
        flags=re.IGNORECASE,
    )
    phrase_text = re.sub(r"^role requiring\s*", "", phrase_text, flags=re.IGNORECASE)
    phrase_text = re.sub(r"^must have\s*", "", phrase_text, flags=re.IGNORECASE)
    return phrase_text


def _extract_skill_phrases(job_description: str) -> list:
    phrase_patterns = [
        r"(?:^|\n)\s*(?:experience with|knowledge of|proficient in|familiar with|strong understanding of|strong knowledge of|experience in|experience supporting|must have|should have|you should have|including|includes|include|skilled in|skilled at|capable of|skills such as|skills include|skills include)\s*[:\-]?\s*([^\n\.]+)",
        r"(?:^|\n)\s*(?:required|desired|preferred|essential|core|key)\s*(?:skills?|qualifications?)\s*[:\-]?\s*([^\n\.]+)",
        r"(?:^|\n)\s*(?:responsibilities|duties|experience)\s*[:\-]?\s*([^\n\.]+)",
    ]
    candidates = []
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, job_description, re.IGNORECASE):
            phrase_text = _clean_skill_phrase(match.group(1).strip())
            candidates.extend(_split_skill_candidates(phrase_text))
    seen = set()
    unique = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate.strip())
    return unique


def extract_job_skills(job_description: str) -> list:
    """Extract required skills from a job description using OpenAI.

    If the OpenAI client is unavailable or the call fails, a robust fallback is used.
    """
    print("========== RAW JD ==========")
    print(job_description)
    prompt = f"""Extract a JSON array of required skills from the following job description.
Strict rules:
1. Extract only skills, technologies, tools, platforms, services, methodologies, databases, programming languages, certifications, or technical requirements that are explicitly stated in the CURRENT JD.
2. Do not infer or invent skills from the job role or general knowledge.
3. Do not convert general troubleshooting language into a different skill such as 'hardware troubleshooting' unless those words or an equivalent explicit hardware requirement appear in the JD.
4. Preserve the original meaning and terminology from the JD.
5. Return ONLY a valid JSON array of strings, with no commentary or explanation.

Job description:\n{job_description}\n"""
    openai_response = ""
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            openai_response = response.choices[0].message.content
        except Exception as e:
            print("OpenAI JD extraction failed:", e)
    else:
        print("OpenAI client not configured – using fallback parser.")
    print("========== OPENAI RAW RESPONSE ==========")
    print(openai_response)

    required = []
    try:
        parsed = json.loads(openai_response)
        if isinstance(parsed, list):
            required = parsed
        else:
            raise ValueError("Not a list")
    except Exception:
        section_text = _extract_job_skill_section(job_description)
        if section_text:
            required = [s for s in _split_skill_candidates(section_text) if s]

        if not required:
            bullet_lines = _extract_skill_lines(job_description)
            if bullet_lines:
                required = [s for s in _split_skill_candidates("\n".join(bullet_lines)) if s]

        if not required:
            required = _extract_skills_by_keyword(job_description)

        if not required:
            required = _extract_skill_phrases(job_description)

        if not required:
            # If there is no explicit skills section, infer from the JD text.
            required = _infer_skills_from_job_description(job_description)

    required = _filter_explicit_skills(_dedupe_skills(required), job_description)

    print("========== EXTRACTED JD SKILLS ==========")
    print(required)
    return required

# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def match_skills(resume_skills: list, jd_skills: list, threshold: float = 0.65):
    """Compare *resume_skills* against *jd_skills* and return (matching, missing).

    Matching is conservative: exact/alias matches are preferred, and semantic similarity is only used
    when the pair is already lexically close and unlikely to be a false positive.
    """
    print("========== RESUME SKILLS ==========")
    print(resume_skills)
    print("========== REQUIRED JD SKILLS ==========")
    print(jd_skills)

    if not resume_skills or not jd_skills:
        if not jd_skills:
            print("JD SKILL EXTRACTION FAILED")
        print("========== MATCHING SKILLS ==========")
        print([])
        print("========== MISSING SKILLS ==========")
        print(jd_skills)
        return [], jd_skills

    resume_norm = [normalize_skill(s) for s in resume_skills]
    jd_norm = [normalize_skill(s) for s in jd_skills]
    print("========== NORMALIZED RESUME SKILLS ==========")
    print(resume_norm)
    print("========== NORMALIZED JD SKILLS ==========")
    print(jd_norm)

    matching = []
    missing = []

    for jd_skill in jd_skills:
        jd_norm_skill = normalize_skill(jd_skill)
        matched = False
        for resume_skill in resume_skills:
            resume_norm_skill = normalize_skill(resume_skill)
            if jd_norm_skill == resume_norm_skill:
                matched = True
                break
            if not jd_norm_skill or not resume_norm_skill:
                continue
            if len(resume_norm_skill.split()) == len(jd_norm_skill.split()) == 1:
                if resume_norm_skill == jd_norm_skill:
                    matched = True
                    break
                if resume_norm_skill in {"js", "javascript"} and jd_norm_skill in {"js", "javascript"}:
                    matched = True
                    break
                if resume_norm_skill in {"node.js", "nodejs", "node js"} and jd_norm_skill in {"node.js", "nodejs", "node js"}:
                    matched = True
                    break
                if resume_norm_skill in {"react", "react.js", "react js"} and jd_norm_skill in {"react", "react.js", "react js"}:
                    matched = True
                    break
                if resume_norm_skill in {"postgresql", "postgres"} and jd_norm_skill in {"postgresql", "postgres"}:
                    matched = True
                    break
                if resume_norm_skill in {"gcp", "google cloud platform", "google cloud"} and jd_norm_skill in {"gcp", "google cloud platform", "google cloud"}:
                    matched = True
                    break
                if resume_norm_skill in {"ci/cd", "ci cd", "cicd"} and jd_norm_skill in {"ci/cd", "ci cd", "cicd"}:
                    matched = True
                    break
                continue

            if SequenceMatcher(None, jd_norm_skill, resume_norm_skill).ratio() >= 0.95:
                matched = True
                break

        if matched:
            matching.append(jd_skill)
        else:
            missing.append(jd_skill)

    if jd_skills and not missing:
        print("SKILL COMPARISON FAILED")
    print("========== MATCHING SKILLS ==========")
    print(matching)
    print("========== MISSING SKILLS ==========")
    print(missing)
    return matching, missing


def get_missing_skills(resume_skills: list, jd_skills: list, threshold: float = 0.65):
    """Convenience wrapper that returns only the missing skills list."""
    _, missing = match_skills(resume_skills, jd_skills, threshold)
    return missing
