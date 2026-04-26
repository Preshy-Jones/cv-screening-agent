from langgraph.graph import StateGraph, END
from models.schemas import AgentState
from agents.jd_analyser import jd_analyser_agent
from agents.cv_parser import cv_parser_agent
from agents.gap_analyser import gap_analyser_agent
from agents.report_writer import report_writer_agent


def create_workflow():
    # Initialise the graph with our state schema
    workflow = StateGraph(AgentState)

    # Add each agent as a node
    workflow.add_node("jd_analyser", jd_analyser_agent)
    workflow.add_node("cv_parser", cv_parser_agent)
    workflow.add_node("gap_analyser", gap_analyser_agent)
    workflow.add_node("report_writer", report_writer_agent)

    # Define the flow — linear pipeline
    workflow.set_entry_point("jd_analyser")
    workflow.add_edge("jd_analyser", "cv_parser")
    workflow.add_edge("cv_parser", "gap_analyser")
    workflow.add_edge("gap_analyser", "report_writer")
    workflow.add_edge("report_writer", END)

    # Compile and return
    return workflow.compile()


# Single instance reused across requests
screening_workflow = create_workflow()