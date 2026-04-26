from pydantic import BaseModel
from typing import Optional

# What flows through the entire graph as state
class AgentState(BaseModel):
    # Inputs
    job_description: str
    cv_text: str

    # Agent 1 output
    jd_analysis: Optional[dict] = None

    # Agent 2 output
    cv_analysis: Optional[dict] = None

    # Agent 3 output
    gap_analysis: Optional[dict] = None

    # Agent 4 output
    final_report: Optional[dict] = None

    # Track errors
    error: Optional[str] = None


class JDAnalysis(BaseModel):
    required_skills: list[str]
    nice_to_have_skills: list[str]
    experience_level: str          # junior / mid / senior / principal
    years_experience_required: Optional[int] = None
    key_responsibilities: list[str]
    tech_stack: list[str]
    domain: str                    # e.g. fintech, healthtech, AI


class CVAnalysis(BaseModel):
    skills: list[str]
    years_experience: int
    recent_stack: list[str]
    ai_experience: list[str]
    projects: list[dict]           # name, description, stack, impact
    education: str
    achievements: list[str]


class GapAnalysis(BaseModel):
    match_score: int               # 0-100
    strong_matches: list[str]      # skills/experience that directly match
    gaps: list[str]                # required skills candidate lacks
    nice_to_have_matches: list[str]
    recommendation: str            # STRONG_APPLY / APPLY / STRETCH / SKIP


class FinalReport(BaseModel):
    match_score: int
    recommendation: str
    summary: str
    what_to_highlight: list[str]   # in cover letter and interview
    gaps_to_address: list[str]     # be honest about or learn quickly
    cv_tailoring_tips: list[str]   # specific tweaks for this role
    suggested_cover_letter_angle: str
