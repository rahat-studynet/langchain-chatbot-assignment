import re

import streamlit as st

from chatbot import build_chain, get_response
from schemas import ChatResponse

CATEGORY_LABELS = {
    "programming": "💻 Programming Assistant",
    "math": "📐 Math Tutor",
    "general": "💬 General Assistant",
}

st.set_page_config(page_title="Smart Router Chatbot", page_icon="🤖")


@st.cache_resource
def load_chain():
    return build_chain()


def format_for_display(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)


def show_response(response: ChatResponse):
    st.caption(
        f"Routed to **{CATEGORY_LABELS[response.category]}** "
        f"· confidence {response.confidence:.0%} · difficulty **{response.difficulty}**"
    )
    st.markdown(format_for_display(response.answer))

    with st.expander("Structured output"):
        st.markdown(f"**Summary:** {format_for_display(response.summary)}")
        st.markdown("**Keywords:** " + ", ".join(f"`{k}`" for k in response.keywords))
        st.markdown("**Follow-up questions:**")
        for q in response.follow_up_questions:
            st.markdown(f"- {q}")
        st.json(response.model_dump())


def show_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_response(ChatResponse(**message["response"]))
        else:
            st.markdown(message["content"])


st.header("🤖 Smart Router Chatbot")
st.write("Built with LangChain RunnableBranch, RunnableParallel and Pydantic structured output.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Chat History")
    if st.button("Clear Chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    user_questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if not user_questions:
        st.write("(empty)")
    for i, q in enumerate(user_questions, start=1):
        st.write(f"{i}. {q}")

try:
    chain = load_chain()
except Exception as e:
    st.error(str(e))
    st.stop()

for message in st.session_state.messages:
    show_message(message)

user_input = st.chat_input("Ask about programming, math or anything else")

if user_input:
    user_message = {"role": "user", "content": user_input}
    show_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_response(chain, user_input, st.session_state.messages)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

    st.session_state.messages.append(user_message)
    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer, "response": response.model_dump()}
    )
    st.rerun()
