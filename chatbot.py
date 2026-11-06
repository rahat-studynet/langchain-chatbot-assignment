import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

from prompts import (
    classifier_prompt,
    general_prompt,
    insights_prompt,
    math_prompt,
    programming_prompt,
    summary_prompt,
)
from schemas import ChatResponse, Insights, QueryCategory

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
HISTORY_LIMIT = 6


def is_category(name):
    return lambda x: x["classification"].category == name


def to_chat_response(result):
    classification = result["classification"]
    insights = result["insights"]
    return ChatResponse(
        category=classification.category,
        confidence=classification.confidence,
        answer=result["answer"],
        summary=result["summary"],
        keywords=insights.keywords,
        follow_up_questions=insights.follow_up_questions,
        difficulty=insights.difficulty,
    )


def build_chain():
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is missing. Add it to your .env file.")

    model = ChatGroq(model=MODEL_NAME, temperature=0.3)
    parser = StrOutputParser()

    classifier_chain = classifier_prompt | model.with_structured_output(QueryCategory)

    branch_chain = RunnableBranch(
        (is_category("programming"), programming_prompt | model | parser),
        (is_category("math"), math_prompt | model | parser),
        general_prompt | model | parser,
    )

    parallel_chain = RunnableParallel(
        {
            "classification": lambda x: x["classification"],
            "answer": lambda x: x["answer"],
            "summary": summary_prompt | model | parser,
            "insights": insights_prompt | model.with_structured_output(Insights),
        }
    )

    return (
        RunnablePassthrough.assign(classification=classifier_chain)
        | RunnablePassthrough.assign(answer=branch_chain)
        | parallel_chain
        | RunnableLambda(to_chat_response)
    )


def format_history(messages):
    recent = messages[-HISTORY_LIMIT:]
    if not recent:
        return "(no previous messages)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def get_response(chain, question, messages):
    return chain.invoke(
        {"question": question, "chat_history": format_history(messages)}
    )
