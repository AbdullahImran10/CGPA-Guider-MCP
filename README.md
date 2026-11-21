# PUCIT GPA Assistant — Assignment 3

**Roll Number:** BCSF24A017

**Name:** Abdullah Imran

## Project Description

The PUCIT GPA Assistant is an AI-powered assistant that helps students calculate semester GPA, projected CGPA, and the GPA required to achieve a target CGPA.

This project uses LangChain, LangGraph, FastMCP, Groq, Streamlit, and SQLite.

## Assignment 3 Implementation

### 1. MCP Server
- Implemented using FastMCP.
- Exposes all 8 GPA-related tools.
- Supports independent tool discovery and execution.

### 2. Agent-MCP Integration
- Uses LangChain MCPAdapter to connect the agent to the MCP server.
- Supports asynchronous tool execution.

### 3. Persistent Memory
- Uses AsyncSqliteSaver for persistent conversation checkpoints.
- Supports thread-based conversations.
- Implements `/switch`, `/history`, and `/forget`.

### 4. Streaming
- Uses LangGraph `astream`.
- Supports `updates` and `messages` streaming modes.
- Displays responses as they are generated.

### 5. Online MCP Server
- Hosted using Prefect Horizon.
- MCP URL:

https://pucit-cgpa-guider.fastmcp.app/mcp

- Authentication: Not required for the current endpoint.

### 6. Server Failure Recovery
- Implements exception handling for MCP connection failures.
- Includes pending tool-call recovery using ToolMessage.

## GitHub Repository

https://github.com/AbdullahImran10/CGPA-Guider-MCP

## Live Application

https://pucit-cgpa-guider.streamlit.app/

## Technologies Used

- Python
- LangChain
- LangGraph
- FastMCP
- Groq
- Streamlit
- SQLite
