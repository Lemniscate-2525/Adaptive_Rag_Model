from src.llms.openai import llm
from pydantic import BaseModel, Field
from typing import Literal


class RouteTest(BaseModel):
    route: Literal["document", "web", "general"] = Field(
        description="The appropriate route."
    )
    reason: str


# Test 1: Basic invocation
print("\n========== BASIC INVOCATION ==========")

response = llm.invoke(
    "Explain what Retrieval-Augmented Generation is in three sentences."
)

print(response.content)

import json

print("\n========== PROMPT-ENFORCED JSON ==========")

response = llm.invoke("""
You are a query classification system.

Classify the following query into exactly one category:
- document
- web
- general

Query: Explain the contents of my uploaded PDF.

Return ONLY a valid JSON object with these fields:
{
    "route": "document",
    "reason": "Brief explanation"
}

Do not include markdown fences or additional text.
""")

print("Raw response:")
print(response.content)

try:
    parsed = json.loads(response.content)
    result = RouteTest.model_validate(parsed)
    print("\nValidated Pydantic object:")
    print(result)
except Exception as e:
    print("\nParsing/validation failed:", repr(e))