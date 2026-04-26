import json
import re
from anthropic import Anthropic
from models.schemas import AgentState, GapAnalysis

client = Anthropic()

def gap_analyser_agent(state: AgentState) -> AgentState:
    print("⚖️  Agent 3: Analysing gaps...")

    prompt = f"""
You are a senior engineering hiring manager. Compare this candidate's 
profile against the job requirements and produce an honest gap analysis.

JOB REQUIREMENTS:
{json.dumps(state.jd_analysis, indent=2)}

CANDIDATE PROFILE:
{json.dumps(state.cv_analysis, indent=2)}

Be honest and specific. Do not inflate the match score.

Return ONLY a JSON object with this exact structure, no other text:
{{
    "match_score": 75,
    "strong_matches": [
        "5+ years Python matches requirement",
        "FastAPI experience directly relevant"
    ],
    "gaps": [
        "No Playwright experience — central to role",
        "No Elixir experience"
    ],
    "nice_to_have_matches": [
        "LLM orchestration experience is a bonus"
    ],
    "recommendation": "APPLY"
}}

Recommendation must be one of:
- STRONG_APPLY (85+ score, most requirements met)
- APPLY (65-84 score, core requirements met, gaps are learnable)
- STRETCH (45-64 score, significant gaps but worth trying)
- SKIP (below 45, too many critical gaps)
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

    gap_data = json.loads(raw)
    GapAnalysis(**gap_data)

    state.gap_analysis = gap_data
    print(f"✅ Gap Analysis complete. "
          f"Score: {gap_data['match_score']}/100, "
          f"Recommendation: {gap_data['recommendation']}")
    return state