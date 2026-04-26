import json
import re
from anthropic import Anthropic
from models.schemas import AgentState, JDAnalysis

client = Anthropic()

def jd_analyser_agent(state: AgentState) -> AgentState:
    print("🔍 Agent 1: Analysing job description...")

    prompt = f"""
You are an expert technical recruiter. Analyse this job description 
and extract structured information.

JOB DESCRIPTION:
{state.job_description}

Return ONLY a JSON object with this exact structure, no other text:
{{
    "required_skills": ["skill1", "skill2"],
    "nice_to_have_skills": ["skill1", "skill2"],
    "experience_level": "senior",
    "years_experience_required": 5,
    "key_responsibilities": ["responsibility1", "responsibility2"],
    "tech_stack": ["Python", "FastAPI"],
    "domain": "fintech"
}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

    jd_data = json.loads(raw)
    jd_data.setdefault("years_experience_required", None)

    # Validate against schema
    JDAnalysis(**jd_data)

    state.jd_analysis = jd_data
    print(f"✅ JD Analysis complete. Domain: {jd_data['domain']}, "
          f"Level: {jd_data['experience_level']}")
    return state
