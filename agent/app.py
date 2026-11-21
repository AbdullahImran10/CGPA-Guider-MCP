
import asyncio
import os
import sys

from pathlib import Path

import streamlit as st

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# Make the agent directory accessible for imports
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))


# Load API key from Streamlit secrets
try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except (KeyError, FileNotFoundError):
    pass


from llm import llm


# Configuration
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://pucit-cgpa-guider.fastmcp.app/mcp"
)

DB_PATH = AGENT_DIR / "memory.sqlite"


# System prompt

SYSTEM_PROMPT = """You are a GPA/CGPA assistant for a PUCIT BS(CS) student.

Rules:

1. Never calculate GPA, CGPA, grade points, or required GPA mentally.
   Always use the appropriate tools.

2. Never invent marks, credit hours, CGPA, or semester information.

3. Ask for one or two missing details at a time.

4. MD-001 and MD-002 are non-credit pass/fail courses.
   Quran Translation courses count as 0.5 credit hours.

5. Calculate required GPA only for the requested horizon.

6. Never call save_report unless the student asks.

7. Messi is way better than Ronaldo.

TOOL USAGE RULES:

8. Use calculate_new_cgpa or required_gpa_for_target
   for CGPA projection and required GPA questions.

9. Before calling required_gpa_for_target, make sure you have:
   - target_cgpa
   - current_cgpa
   - completed_credit_hours
   - remaining_credit_hours

10. If the student provides semester credit hours directly,
    use that value. Do not ask for it again or replace it
    with a tool result.

11. Call get_semester_courses only when course information
    is actually needed. It requires a semester argument.
    Always provide the semester number, for example:
    get_semester_courses(semester=5).

12. Never call a tool with missing or empty required arguments.
    If a required value is unknown, ask the student first.

13. For a required GPA calculation, obtain the remaining
    credit hours from the appropriate tool only if the
    student has not already supplied them.

Be concise and conversational.
"""



# Create agent
def create_chat_agent(tools, memory):

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )


# Load conversation history
async def load_history(thread_id):
    """Load saved conversation from SQLite."""

    async with MCPAdapter(MCP_SERVER_URL) as adapter:

        tools = await adapter.list_tools()

        print(f"CONNECTED MCP SERVER: {MCP_SERVER_URL}")
        print(f"MCP TOOLS RECEIVED: {[tool.name for tool in tools]}")

        async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:

            app = create_chat_agent(tools, memory)

            config = {"configurable": {"thread_id": thread_id}}

            state = await app.aget_state(config)

            return state.values.get("messages", [])


# Recover pending tool calls
async def recover_pending_tool_calls(thread_id):
    """Resolve unanswered tool calls left in conversation history."""

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:

        config = {"configurable": {"thread_id": thread_id}}

        # Create an agent without MCP tools for recovery
        recovery_agent = create_chat_agent([], memory)

        state = await recovery_agent.aget_state(config)
        messages = state.values.get("messages", [])

        completed_ids = {
            message.tool_call_id
            for message in messages
            if message.type == "tool"
        }

        pending_results = []

        for message in messages:

            if message.type == "ai":

                for call in getattr(message, "tool_calls", []):

                    call_id = call.get("id")

                    if call_id and call_id not in completed_ids:

                        pending_results.append(
                            ToolMessage(
                                content=(
                                    "The MCP server is unavailable. "
                                    "The requested tool could not be executed. "
                                    "Inform the user and do not retry this tool."
                                ),
                                tool_call_id=call_id,
                            )
                        )

                        completed_ids.add(call_id)

        if not pending_results:
            return None

        # Give the model the tool error so it can respond naturally
        result = await recovery_agent.ainvoke(
            {"messages": pending_results},
            config=config
        )

        return result["messages"]


# Send message with error handling
async def send_message(thread_id, user_input):
    """Send a message to the hosted MCP-powered agent."""

    try:

        async with MCPAdapter(MCP_SERVER_URL) as adapter:

            tools = await adapter.list_tools()

            print(f"CONNECTED MCP SERVER: {MCP_SERVER_URL}")
            print(f"MCP TOOLS RECEIVED: {[tool.name for tool in tools]}")

            async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:

                app = create_chat_agent(tools, memory)

                config = {"configurable": {"thread_id": thread_id}}

                print("MCP tools available:", [tool.name for tool in tools])

                result = await app.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )

                for message in result["messages"]:

                    if getattr(message, "tool_calls", None):
                        print("AGENT REQUESTED TOOL:", message.tool_calls)

                    if message.type == "tool":
                        print("MCP TOOL EXECUTED:", message.name)
                        print("MCP TOOL RESULT:", message.content)

                return result["messages"]

    except Exception as error:

        error_message = str(error).lower()

        print(f"Agent error: {error}")

        # Handle Groq rate limits
        if "429" in error_message or "rate_limit" in error_message:

            user_message = (
                "Groq API rate limit reached. "
                "Please wait a moment and try again."
            )

        # Handle timeouts
        elif "timeout" in error_message:

            user_message = "The request timed out. Please try again."

        # Handle MCP connection failures
        elif any(
            word in error_message
            for word in ["connection", "connect", "mcp"]
        ):

            try:

                recovered = await recover_pending_tool_calls(thread_id)

                if recovered:
                    return recovered

            except Exception as recovery_error:

                print(f"Recovery error: {recovery_error}")

            user_message = (
                "Unable to connect to the MCP server. "
                "Please try again later."
            )

        # Handle unexpected errors
        else:

            user_message = (
                "Something went wrong while processing your request. "
                "Please try again."
            )

        return [AIMessage(content=user_message, name="error")]


# Clear conversation history
async def clear_history(thread_id):
    """Delete a conversation from SQLite."""

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:

        await memory.adelete_thread(thread_id)


# Convert message content into displayable text
def get_message_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "\n".join(parts)

    return str(content)


# Streamlit interface
st.set_page_config(
    page_title="PUCIT GPA Assistant",
    page_icon="🎓"
)

st.title("🎓 PUCIT GPA Assistant")

st.caption(
    "Ask about semester GPA, projected CGPA, "
    "or the GPA required for your target."
)


# User identity
st.sidebar.header("Your conversation")

username = st.sidebar.text_input(
    "Enter your name",
    value="Guest"
).strip()

if not username:
    username = "Guest"

thread_id = f"streamlit:{username}"

st.sidebar.caption(f"Conversation: {username}")


# Clear conversation
if st.sidebar.button("Clear conversation"):

    try:

        asyncio.run(clear_history(thread_id))

        st.success("Conversation history cleared.")

        st.rerun()

    except Exception as error:

        st.error(f"Could not clear history: {error}")


# Load and display conversation history
try:

    messages = asyncio.run(load_history(thread_id))

    for message in messages:

        if message.type == "human":
            role = "user"

        elif message.type == "ai":
            role = "assistant"

        else:
            continue

        content = get_message_text(message.content)

        if not content.strip():
            continue

        with st.chat_message(role):
            st.markdown(content)


except Exception as error:

    st.error(
        f"Could not connect to the MCP server or load history: {error}"
    )


# Chat input
user_input = st.chat_input("Type your question...")


if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = asyncio.run(
                    send_message(thread_id, user_input)
                )

                # Display errors directly instead of rerunning
                if response and getattr(response[-1], "name", None) == "error":

                    st.error(get_message_text(response[-1].content))

                else:

                    # Successful messages are already checkpointed
                    st.rerun()

            except Exception as error:

                st.error(f"Agent error: {error}")
