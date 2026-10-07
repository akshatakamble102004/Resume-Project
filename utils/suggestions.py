
# utils/suggestions.py

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    # pyrefly: ignore [missing-import]
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

client = None
if OpenAI is not None:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_suggestions(match_score, missing_skills, ats_feedback):
    """Rule-based version — consistent, context-aware, non-contradictory."""
    suggestions = []
    # Debug logging of inputs
    print("RESUME MATCH SCORE:", match_score)
    print("MISSING SKILLS:", missing_skills)
    print("ATS FEEDBACK:", ats_feedback)

    has_gaps = bool(missing_skills)
    HIGH_THRESHOLD = 70  # match_score is a 0–1 float from the model
    # match_score comes in as a 0–1 float (e.g. 0.66), but the route
    # multiplies by 100 only for display — we compare against the raw value here.
    # Guard: if caller already passes a percentage (>1), normalise it.
    score = match_score if match_score <= 1.0 else match_score / 100.0

    if has_gaps:
        # ── Case: skill gaps exist ──────────────────────────────────────────
        skill_list = ", ".join(missing_skills)
        if score < (HIGH_THRESHOLD / 100):
            suggestions.append(
                "Your resume has some skill gaps for this job. "
                "Consider tailoring your resume to the job requirements."
            )
        else:
            suggestions.append(
                "Your resume aligns well overall, but a few required skills are missing."
            )
        suggestions.append(
            f"Consider adding these skills if you have them: {skill_list}."
        )
    else:
        # ── Case: no skill gaps ─────────────────────────────────────────────
        if score >= (HIGH_THRESHOLD / 100):
            # Good score AND no gaps — genuinely strong match
            suggestions.append(
                "Great — your resume covers all key skills mentioned in the job description."
            )
            suggestions.append(
                "Strong match! Fine-tune a few details (e.g. quantified achievements, "
                "job-specific keywords) to make it even stronger."
            )
        else:
            # No gaps but score is lower — alignment is content-level, not skill-level
            suggestions.append(
                "Great — your resume covers all key skills mentioned in the job description."
            )
            suggestions.append(
                "Your required skills are covered, but the overall resume-to-job alignment "
                "could be improved. Consider enhancing:"
            )
            suggestions.append(
                "• Project and experience descriptions — highlight responsibilities "
                "that mirror the job requirements."
            )
            suggestions.append(
                "• Measurable achievements — use numbers and outcomes to demonstrate impact."
            )
            suggestions.append(
                "• Job-specific keywords — naturally incorporate terminology from the job "
                "description into your experience section."
            )

    if ats_feedback:
        suggestions.extend(ats_feedback)

    return suggestions


def generate_suggestions_llm(match_score, missing_skills, ats_feedback):
    """LLM version — more natural-sounding, uses OpenAI GPT when available."""
    if client is None:
        return generate_suggestions(match_score, missing_skills, ats_feedback)

    prompt = f"""Resume match score: {match_score}%
Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}
ATS issues: {', '.join(ats_feedback) if ats_feedback else 'None'}

Write 3 short, specific, encouraging suggestions to help this candidate improve their resume for this job. Return them as a numbered list."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        suggestions = [line.strip() for line in text.split("\n") if line.strip()]
        return suggestions
    except Exception as e:
        print(f"LLM call failed: {e}")
        return generate_suggestions(match_score, missing_skills, ats_feedback)