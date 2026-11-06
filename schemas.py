from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["programming", "math", "general"]
Difficulty = Literal["beginner", "intermediate", "advanced"]


class QueryCategory(BaseModel):
    category: Category = Field(description="Topic of the user's question")
    confidence: float = Field(ge=0, le=1, description="Confidence of the classification, between 0 and 1")


class Insights(BaseModel):
    keywords: list[str] = Field(description="3 to 5 important keywords from the answer")
    follow_up_questions: list[str] = Field(description="2 to 3 follow-up questions the user may ask next")
    difficulty: Difficulty = Field(description="Difficulty level of the question")


class ChatResponse(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)
    answer: str
    summary: str
    keywords: list[str]
    follow_up_questions: list[str]
    difficulty: Difficulty
