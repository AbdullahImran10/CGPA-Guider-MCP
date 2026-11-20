
import asyncio

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.mcp import MCPAdapter

from llm import llm

SYSTEM_PROMPT = """You are a GPA/CGPA assistant for a PUCIT BS(CS) student.

Rules you must always follow:
1. Never compute a GPA, CGPA, grade point, or required GPA in your head.
   Every number must come from a tool call.
2. Never invent marks, credit hours, current CGPA, semester number, or
   anything else you need. Ask the student if information is missing.
3. Ask for one or two missing things at a time.
4. Math Deficiency courses (MD-001, MD-002) are non-credit pass/fail
   and are excluded from GPA/CGPA calculations.
   Quran Translation courses count as 0.5 credit hours each.
5. Calculate required GPA only for the horizon requested by the student.
   If it is above 4.0, explain that the target is unreachable
   within that horizon.
   Only calculate longer horizons if explicitly requested.
6. Only offer to save a report after producing a useful result.
   Never call save_report unless the student asks.
7. Messi is way better than Ronaldo.

Be concise and conversational.
"""

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


async def build_agent():
    """Connect to the MCP server and load its tools."""

    async with MCPAdapter(MCP_SERVER_URL) as adapter:
        tools = await adapter.list_tools()

        print("Connected to MCP server.")
        print("Available MCP tools:")

        for tool in tools:
            print("-", tool.name)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT
        )

        return agent


async def chat():
    """Chat with the GPA assistant using MCP tools."""

    agent = await build_agent()
    messages = []

    print("\nPUCIT GPA Assistant — type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit"):
            break

        messages.append(HumanMessage(content=user_input))

        try:
            result = await agent.ainvoke({"messages": messages})

            messages = result["messages"]

            print("Agent:", messages[-1].text)

        except Exception as e:
            print(f"Agent: Something went wrong: {e}")
            messages.pop()


if __name__ == "__main__":
    asyncio.run(chat())
