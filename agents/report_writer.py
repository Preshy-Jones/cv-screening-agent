import json
import re
from anthropic import Anthropic
from models.schemas import AgentState, FinalReport

client = Anthropic()

def report_writer_agent(state: AgentState) -> AgentState:
    print("📝 Agent 4: Writing final report...")

    prompt = f"""
You are a career coach specialising in tech roles. 
Write an actionable application report for this candidate.

JOB REQUIREMENTS:
{json.dumps(state.jd_analysis, indent=2)}

CANDIDATE PROFILE:
{json.dumps(state.cv_analysis, indent=2)}

GAP ANALYSIS:
{json.dumps(state.gap_analysis, indent=2)}

Return ONLY a JSON object with this exact structure, no other text:
{{
    "match_score": 75,
    "recommendation": "APPLY",
    "summary": "2-3 sentence honest summary of fit",
    "what_to_highlight": [
        "Lead with BandReady — shows end-to-end AI product ownership",
        "Mention PalmPay scale — 10M users signals production reliability"
    ],
    "gaps_to_address": [
        "Playwright: acknowledge you're picking it up, show initiative",
        "Elixir: they said it's a plus, don't stress it"
    ],
    "cv_tailoring_tips": [
        "Move AI products section above employment history",
        "Add browser automation to project descriptions where relevant"
    ],
    "suggested_cover_letter_angle": "1-2 sentences on the exact angle 
     to lead with in the cover letter"
}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

    report_data = json.loads(raw)
    FinalReport(**report_data)

    state.final_report = report_data
    print(f"✅ Report complete. Recommendation: {report_data['recommendation']}")
    return state