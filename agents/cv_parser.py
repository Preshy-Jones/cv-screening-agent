import json
import re
from anthropic import Anthropic
from models.schemas import AgentState, CVAnalysis

client = Anthropic()

def cv_parser_agent(state: AgentState) -> AgentState:
    print("📄 Agent 2: Parsing CV...")

    prompt = f"""
You are an expert CV analyser. Extract structured information 
from this CV/resume.

CV TEXT:
{state.cv_text}

Return ONLY a JSON object with this exact structure, no other text:
{{
    "skills": ["Python", "FastAPI", "LangChain"],
    "years_experience": 6,
    "recent_stack": ["Python", "TypeScript", "AWS"],
    "ai_experience": ["Built LLM scoring engine", "RAG pipelines"],
    "projects": [
        {{
            "name": "BandReady",
            "description": "AI IELTS prep platform",
            "stack": ["Python", "Claude API"],
            "impact": "Live product with paying users"
        }}
    ],
    "education": "B.Eng Mechanical Engineering, University of Lagos",
    "achievements": ["$3M transaction volume at SkyeWallet"]
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

    cv_data = json.loads(raw)
    CVAnalysis(**cv_data)

    state.cv_analysis = cv_data
    print(f"✅ CV Analysis complete. "
          f"Skills found: {len(cv_data['skills'])}, "
          f"Experience: {cv_data['years_experience']} years")
    return state