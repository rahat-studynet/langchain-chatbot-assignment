from langchain_core.prompts import PromptTemplate

classifier_prompt = PromptTemplate(
    template="""
    Classify the user's latest question into exactly one category:
    - programming : coding, software, debugging, algorithms, tools
    - math : arithmetic, algebra, calculus, statistics, math problems
    - general : anything else

    Conversation so far:
    {chat_history}

    Latest question: {question}
    """,
    input_variables=["question", "chat_history"],
)

programming_prompt = PromptTemplate(
    template="""
    You are an expert Programming Assistant.
    Answer the question clearly, include a short code example when it helps,
    and explain the code briefly. Keep the answer concise (under 250 words).

    Conversation so far:
    {chat_history}

    Question: {question}
    """,
    input_variables=["question", "chat_history"],
)

math_prompt = PromptTemplate(
    template="""
    You are a patient Math Tutor.
    Solve the problem step by step, show every calculation,
    and clearly state the final answer at the end. Keep it concise.
    Write math using $...$ for inline and $$...$$ for block equations.

    Conversation so far:
    {chat_history}

    Question: {question}
    """,
    input_variables=["question", "chat_history"],
)

general_prompt = PromptTemplate(
    template="""
    You are a friendly General Assistant.
    Give an accurate, well-structured and easy to understand answer.
    Keep the answer concise (under 250 words).

    Conversation so far:
    {chat_history}

    Question: {question}
    """,
    input_variables=["question", "chat_history"],
)

summary_prompt = PromptTemplate(
    template="""
    Summarize the following answer in 2 short sentences:
    {answer}
    """,
    input_variables=["answer"],
)

insights_prompt = PromptTemplate(
    template="""
    Based on the question and answer below, extract the key keywords,
    suggest follow-up questions and rate the difficulty level.

    Question: {question}
    Answer: {answer}
    """,
    input_variables=["question", "answer"],
)
