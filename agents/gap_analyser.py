import json
import re
from anthropic import Anthropic
from openai import OpenAI
from models.schemas import AgentState, GapAnalysis

anthropic_client = Anthropic()
openai_client = OpenAI()

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a ** 2 for a in vec_a) ** 0.5
    mag_b = sum(b ** 2 for b in vec_b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def keyword_match(required: str, cv_skills: list[str], cv_text: str) -> bool:
    """
    Check if any word from the required skill appears anywhere
    in the CV skills list or raw CV text.
    Fast, free, no API call needed.
    """
    # Extract meaningful words (ignore common words)
    stop_words = {"and", "or", "the", "a", "an", "with", "for", "of", "in"}
    
    required_words = [
        w.lower().strip("(),")
        for w in required.split()
        if w.lower() not in stop_words and len(w) > 2
    ]
    
    cv_text_lower = cv_text.lower()
    cv_skills_lower = " ".join(cv_skills).lower()
    
    for word in required_words:
        if word in cv_text_lower or word in cv_skills_lower:
            return True
    
    return False

async def gap_analyser_agent(state: AgentState) -> AgentState:
    print("⚖️  Agent 3: Analysing gaps...")

    required_skills = state.jd_analysis["required_skills"]
    cv_skills = state.cv_analysis["skills"]
    cv_text = state.cv_text

    # Embed all required skills — one API call
    req_response = openai_client.embeddings.create(
        input=required_skills,
        model="text-embedding-3-small"
    )
    req_vectors = [item.embedding for item in req_response.data]

    # Embed all CV skills — one API call
    cv_response = openai_client.embeddings.create(
        input=cv_skills,
        model="text-embedding-3-small"
    )
    cv_vectors = [item.embedding for item in cv_response.data]

    gaps = []
    strong_matches = []

    for i, required_skill in enumerate(required_skills):
        req_vec = req_vectors[i]

        # STEP 1 — keyword check first (free, fast)
        if keyword_match(required_skill, cv_skills, cv_text):
            strong_matches.append(
                f"{required_skill} — keyword match in CV"
            )
            continue

        # STEP 2 — semantic similarity (catches paraphrasing)
        best_score = 0
        best_cv_skill = None

        for j, cv_skill in enumerate(cv_skills):
            score = cosine_similarity(req_vec, cv_vectors[j])
            if score > best_score:
                best_score = score
                best_cv_skill = cv_skill

        if best_score > 0.72:  # lowered from 0.82
            strong_matches.append(
                f"{required_skill} — matched via '{best_cv_skill}' "
                f"(similarity: {best_score:.2f})"
            )
        else:
            gaps.append(required_skill)

    match_score = int(len(strong_matches) / len(required_skills) * 100)

    gap_data = {
        "match_score": match_score,
        "strong_matches": strong_matches,
        "gaps": gaps,
        "nice_to_have_matches": [],
        "recommendation": (
            "STRONG_APPLY" if match_score >= 85 else
            "APPLY"        if match_score >= 65 else
            "STRETCH"      if match_score >= 45 else
            "SKIP"
        )
    }

    GapAnalysis(**gap_data)

    state.gap_analysis = gap_data
    print(
        f"✅ Gap Analysis complete. "
        f"Score: {gap_data['match_score']}/100, "
        f"Recommendation: {gap_data['recommendation']}"
    )
    return state