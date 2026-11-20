
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

# Make the agent folder accessible
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools import (
    marks_to_grade_points,
    calculate_semester_gpa,
    calculate_new_cgpa,
    required_gpa_for_target,
    get_semester_courses,
    get_credit_hours_after_semester,
    get_remaining_credit_hours,
    save_report,
)

# Create MCP server
mcp = FastMCP("PUCIT-CGPA-Guider")

# Register all 8 tools
TOOLS = [
    marks_to_grade_points,
    calculate_semester_gpa,
    calculate_new_cgpa,
    required_gpa_for_target,
    get_semester_courses,
    get_credit_hours_after_semester,
    get_remaining_credit_hours,
    save_report,
]

for langchain_tool in TOOLS:
    mcp.tool(langchain_tool.func)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
