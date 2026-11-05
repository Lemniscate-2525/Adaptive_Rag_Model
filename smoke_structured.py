
from pydantic import BaseModel, Field
from typing import Literal

from src.llms.structured import invoke_structured

class RouteTest(BaseModel):
    route: Literal["index", "general", "search"] = Field(
        description="The appropriate query route."
    )


class GradeTest(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the retrieved context is relevant."
    )


print("\n========== TEST 1: QUERY CLASSIFICATION ==========")

route = invoke_structured(
    prompt="""
    Classify this query:
    Explain the contents of my uploaded PDF.
    Choose index, general, or search.
    """,
    schema=RouteTest,
)

print(route)
print("Type:", type(route))


print("\n========== TEST 2: DOCUMENT GRADING ==========")

grade = invoke_structured(
    prompt="""
    Question: What is photosynthesis?

    Context:
    Photosynthesis is the process by which green plants
    convert sunlight, water and carbon dioxide into glucose
    and oxygen.
    """,
    schema=GradeTest,
)

print(grade)
print("Type:", type(grade))
