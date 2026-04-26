import os
from typing import Any, Union
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

load_dotenv()

from models.schemas import AgentState
from graph.workflow import screening_workflow
from utils.pdf_extractor import extract_text_from_bytes

app = FastAPI(
    title="CV Screening Agent",
    description="Multi-agent CV screening powered by LangGraph + Claude",
    version="1.0.0"
)


def _state_to_dict(state: Union[AgentState, dict]) -> dict:
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state)  # Fallback for non-Pydantic mapping-like state objects


@app.get("/")
def root():
    return {"status": "running", "agents": 4}


@app.post("/screen")
async def screen_cv(
    job_description: str = Form(...),
    cv_file: UploadFile = File(...)
):
    # Validate file type
    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    # Extract CV text from uploaded PDF
    cv_bytes = await cv_file.read()
    cv_text = extract_text_from_bytes(cv_bytes)

    if len(cv_text.strip()) < 100:
        raise HTTPException(400, "Could not extract enough text from PDF")

    # Build initial state
    initial_state = AgentState(
        job_description=job_description,
        cv_text=cv_text
    )

    # Run through all 4 agents
    print("\n🚀 Starting CV screening workflow...\n")
    final_state = screening_workflow.invoke(initial_state)
    final_state_data = _state_to_dict(final_state)
    print("\n✅ Workflow complete!\n")

    return JSONResponse(content={
        "success": True,
        "report": final_state_data["final_report"],
        "detailed": {
            "jd_analysis": final_state_data["jd_analysis"],
            "cv_analysis": final_state_data["cv_analysis"],
            "gap_analysis": final_state_data["gap_analysis"]
        }
    })


@app.post("/screen/text")
async def screen_cv_text(payload: dict):
    """Alternative endpoint for plain text CV input"""
    job_description = payload.get("job_description")
    cv_text = payload.get("cv_text")

    if not job_description or not cv_text:
        raise HTTPException(400, "Both job_description and cv_text required")

    initial_state = AgentState(
        job_description=job_description,
        cv_text=cv_text
    )

    print("\n🚀 Starting CV screening workflow...\n")
    final_state = screening_workflow.invoke(initial_state)
    final_state_data = _state_to_dict(final_state)
    print("\n✅ Workflow complete!\n")

    return JSONResponse(content={
        "success": True,
        "report": final_state_data["final_report"]
    })
