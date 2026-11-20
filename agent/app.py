import streamlit as st
from langchain.messages import HumanMessage

from agent import agent

st.set_page_config(page_title="PUCIT GPA Assistant", page_icon="🎓")
st.title("🎓 PUCIT GPA Assistant")
st.caption("Ask about a semester GPA, a projected CGPA, or what GPA you need for a target.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.text)

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": st.session_state.messages})
                st.session_state.messages = result["messages"]
                reply = st.session_state.messages[-1].text
            except Exception as e:
                # drop the message that triggered the failed turn so the
                # user can retry without it poisoning future turns
                st.session_state.messages.pop()
                reply = f"Sorry, something went wrong on that turn ({e}). Try rephrasing."
        st.markdown(reply)
