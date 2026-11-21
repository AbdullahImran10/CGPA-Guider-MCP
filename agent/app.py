
import asyncio
import os
import sys
import subprocess
import socket
import time

from pathlib import Path

import streamlit as st

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
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
    "http://127.0.0.1:8000/mcp"
)

DB_PATH = AGENT_DIR / "memory.sqlite"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = PROJECT_ROOT / "server" / "server.py"
MCP_PORT = 8000


# Start MCP server
@st.cache_resource
def start_mcp_server():
    """Start the MCP server once for this Streamlit process."""

    env = os.environ.copy()
    env["PORT"] = str(MCP_PORT)

    process = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):

        if process.poll() is not None:
            raise RuntimeError("MCP server stopped unexpectedly.")

        try:
            with socket.create_connection(
                ("127.0.0.1", MCP_PORT),
                timeout=1
            ):
                return process

        except OSError:
            time.sleep(1)

    process.terminate()

    raise RuntimeError(
        "MCP server did not start within 30 seconds."
    )


start_mcp_server()


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

        async with AsyncSqliteSaver.from_conn_string(
            str(DB_PATH)
        ) as memory:

            app = create_chat_agent(tools, memory)

            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            state = await app.aget_state(config)

            return state.values.get("messages", [])


# Send message with error handling
async def send_message(thread_id, user_input):
    """Send a message to the MCP-powered agent."""

    try:

        # TEMPORARY TEST:
        # Uncomment this line to simulate a 429 error.
        # raise Exception("429 rate_limit_exceeded")

        async with MCPAdapter(MCP_SERVER_URL) as adapter:

            tools = await adapter.list_tools()

            async with AsyncSqliteSaver.from_conn_string(
                str(DB_PATH)
            ) as memory:

                app = create_chat_agent(tools, memory)

                config = {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }

                print(
                    "MCP tools available:",
                    [tool.name for tool in tools]
                )

                result = await app.ainvoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=config
                )

                for message in result["messages"]:

                    if getattr(message, "tool_calls", None):
                        print("Tool calls:", message.tool_calls)

                    if message.type == "tool":
                        print("Tool result:", message.content)

                return result["messages"]

    except Exception as error:

        error_message = str(error).lower()

        print(f"Agent error: {error}")

        if "429" in error_message or "rate_limit" in error_message:

            user_message = (
                "Groq API rate limit reached. "
                "Please wait a moment and try again."
            )

        elif "timeout" in error_message:

            user_message = (
                "The request timed out. Please try again."
            )

        elif any(
            word in error_message
            for word in ["connection", "connect", "mcp"]
        ):

            user_message = (
                "Unable to connect to the MCP server. "
                "Please try again later."
            )

        else:

            user_message = (
                "Something went wrong while processing your request. "
                "Please try again."
            )

        return [
            AIMessage(
                content=user_message,
                name="error"
            )
        ]


# Clear conversation history
async def clear_history(thread_id):
    """Delete a conversation from SQLite."""

    async with AsyncSqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as memory:

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
                if (
                    response
                    and getattr(response[-1], "name", None) == "error"
                ):

                    st.error(
                        get_message_text(response[-1].content)
                    )

                else:

                    # Successful messages are already checkpointed
                    st.rerun()

            except Exception as error:

                st.error(f"Agent error: {error}")
