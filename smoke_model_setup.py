import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------
# TEST 1: OpenRouter connectivity
# --------------------------------

print("\n[1] Testing OpenRouter...")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

response = llm.invoke("Reply with exactly: OpenRouter connected.")

print("Response:", response.content)


# --------------------------------
# TEST 2: Structured output
# --------------------------------

print("\n[2] Testing Gemma structured output...")

from pydantic import BaseModel, Field
from typing import Literal


class RouteTest(BaseModel):
    route: Literal["document", "web", "general"] = Field(
        description="Choose the most appropriate route."
    )


structured_llm = llm.with_structured_output(RouteTest)

result = structured_llm.invoke(
    "Classify this query: What is the capital of France? "
    "Return general."
)

print("Structured result:", result)


# --------------------------------
# TEST 3: Local embeddings
# --------------------------------

print("\n[3] Testing Hugging Face embeddings...")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector = embeddings.embed_query("Testing local embeddings.")

print("Embedding dimensions:", len(vector))
print("First five values:", vector[:5])

print("\nAll integration tests completed.")