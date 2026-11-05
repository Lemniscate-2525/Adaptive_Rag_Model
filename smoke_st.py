
from typing import Literal

from pydantic import BaseModel, Field

from src.llms.structured import invoke_structured


class RouteDecision(BaseModel):
    route: Literal["vectorstore", "web_search", "general"]
    reasoning: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


if __name__ == "__main__":
    result = invoke_structured(
        prompt=(
            "A user asks: 'What is the latest NVIDIA GPU architecture "
            "announced by NVIDIA?' This requires current external information. "
            "Classify the appropriate route."
        ),
        schema=RouteDecision,
        max_attempts=3,
    )

    print("\nStructured output:")
    print(result.model_dump_json(indent=2))

    print("\nValidation:")
    print("Pydantic validation passed.")
