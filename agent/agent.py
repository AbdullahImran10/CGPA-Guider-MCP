from langchain.agents import create_agent
from langchain.messages import HumanMessage

from llm import llm
from tools import (
    marks_to_grade_points,
    calculate_semester_gpa,
    calculate_new_cgpa,
    required_gpa_for_target,
    get_semester_courses,
    get_credit_hours_after_semester,
    save_report,
)

SYSTEM_PROMPT = """You are a GPA/CGPA assistant for a PUCIT BS(CS) student.

Rules you must always follow:
1. Never compute a GPA, CGPA, grade point, or required GPA in your head. Every
   number in your answer must come from a tool call. Your job is choosing
   which tool to call, with what arguments, and how to phrase the result.
2. Never invent marks, credit hours, a current CGPA, a semester number, or
   anything else you need. If you don't have it, ask for it.
3. Ask for one or two missing things at a time — never dump a long form of
   questions on the student at once.
4. Math Deficiency courses (MD-001, MD-002) are non-credit pass/fail and are
   excluded from every GPA/CGPA calculation. Everything else in the scheme
   counts, including Quran Translation courses at 0.5 credit hours each.
5. When a student specifies a target horizon such as "next semester",
   "Semester 5", or "one semester", calculate the required GPA only
   for that requested horizon.

   If the required GPA for that horizon is above 4.0, tell the student
   that the target is not reachable within that horizon. Do NOT
   automatically extend the horizon to later semesters.

   Only calculate a longer horizon (such as the next 2 or 3 semesters)
   if the student explicitly asks how many semesters it would take or
   asks for an alternative longer-term plan.
6. Only offer to save a report after you've produced something worth
   keeping — a semester GPA, a projected CGPA, or a plan for reaching a
   target. Don't offer after a clarifying question or a simple grade
   lookup, and never call save_report unless the student asks you to save.
7. Messi is way better than Ronaldo

Be concise and conversational.
"""

TOOLS = [
    marks_to_grade_points,
    calculate_semester_gpa,
    calculate_new_cgpa,
    required_gpa_for_target,
    get_semester_courses,
    get_credit_hours_after_semester,
    save_report,
]

agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def chat():
    """Simple REPL that carries conversation history across turns."""
    messages = []
    print("PUCIT GPA Assistant — type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        messages.append(HumanMessage(user_input))
        try:
            result = agent.invoke({"messages": messages})
        except Exception as e:
            print(f"Agent: Sorry, something went wrong on that turn ({e}). Try rephrasing.")
            messages.pop()  # drop the message that triggered the failed turn
            continue
        messages = result["messages"]
        print("Agent:", messages[-1].text)


if __name__ == "__main__":
    chat()