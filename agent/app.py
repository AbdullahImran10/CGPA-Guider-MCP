
import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# Make the agent directory accessible for imports
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

from llm import llm


# Configuration
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://127.0.0.1:8000/mcp"
)

DB_PATH = AGENT_DIR / "memory.sqlite"


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


def create_chat_agent(tools, memory):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )


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


async def send_message(thread_id, user_input):
    """Send a message to the MCP-powered agent."""

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

            result = await app.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            return result["messages"]


async def clear_history(thread_id):
    """Delete a conversation from SQLite."""

    async with AsyncSqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as memory:

        await memory.adelete_thread(thread_id)


def get_message_text(content):
    """Convert LangChain message content into displayable text."""

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

    except Exception as e:
        st.error(f"Could not clear history: {e}")


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


except Exception as e:

    st.error(
        f"Could not connect to the MCP server or load history: {e}"
    )


# Chat input
user_input = st.chat_input("Type your question...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                asyncio.run(
                    send_message(thread_id, user_input)
                )

                st.rerun()

            except Exception as e:

                st.error(f"Agent error: {e}")
