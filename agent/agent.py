
import asyncio
from pathlib import Path
import os
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://127.0.0.1:8000/mcp"
)

DB_PATH = Path(__file__).resolve().parent / "memory.sqlite"


async def list_threads(memory):
    """Return all saved conversation thread IDs."""

    threads = []

    async for checkpoint in memory.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id not in threads:
            threads.append(thread_id)

    return threads


async def show_history(app, config):
    """Display conversation history."""

    state = await app.aget_state(config)
    messages = state.values.get("messages", [])

    if not messages:
        print("No conversation history found.")
        return

    print("\n--- Conversation History ---")

    for message in messages:
        print(f"{message.type.upper()}: {message.content}")

    print("----------------------------")


async def main():

    # Connect to MCP server
    async with MCPAdapter(MCP_SERVER_URL) as adapter:

        tools = await adapter.list_tools()

        print("Connected to MCP server.")

        # Open SQLite memory
        async with AsyncSqliteSaver.from_conn_string(
            str(DB_PATH)
        ) as memory:

            # Create agent with persistent memory
            app = create_agent(
                model=llm,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                checkpointer=memory
            )

            # Show saved conversations
            saved_chats = await list_threads(memory)

            print("\nSaved conversations:")

            if saved_chats:
                for chat_name in saved_chats:
                    print("-", chat_name)
            else:
                print("No saved conversations yet.")

            # Select user
            name = input("\nWho is chatting? ").strip()

            while not name:
                name = input("Please enter a name: ").strip()

            config = {
                "configurable": {
                    "thread_id": name
                }
            }

            print(f"\nWelcome, {name}!")
            print("Commands: /switch, /history, /forget, /quit")

            # Chat loop
            while True:

                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("/quit", "quit", "exit"):
                    print("Goodbye!")
                    break

                elif user_input.lower() == "/switch":

                    name = input("Enter the other person's name: ").strip()

                    if not name:
                        print("Name cannot be empty.")
                        continue

                    config = {
                        "configurable": {
                            "thread_id": name
                        }
                    }

                    print(f"Switched to {name}'s conversation.")

                elif user_input.lower() == "/history":

                    await show_history(app, config)

                elif user_input.lower() == "/forget":

                    await memory.adelete_thread(name)

                    print(f"Conversation history for {name} deleted.")

                else:

                    try:
                        print("Agent: ", end="", flush=True)

                        async for message, metadata in app.astream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                            stream_mode="messages"
                        ):

                            # Accept AI messages and AI message chunks
                            if message.type not in ("ai", "AIMessageChunk"):
                                continue

                            content = message.content

                            if isinstance(content, str):
                                print(content, end="", flush=True)

                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        print(
                                            block.get("text", ""),
                                            end="",
                                            flush=True
                                        )

                        print()

                    except Exception as e:
                        print(f"\nAgent error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
