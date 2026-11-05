
"""Prompt-enforced structured output with Pydantic validation."""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.llms.openai import llm

T = TypeVar("T", bound=BaseModel)


def _extract_json_object(content: str) -> dict:
    """Extract the first valid JSON object from an LLM response."""

    text = content.strip()

    # Handle Markdown code fences.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    decoder = json.JSONDecoder()

    # Find a valid JSON object, even if surrounded by commentary.
    for position, character in enumerate(text):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(text[position:])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in model response.")


def invoke_structured(
    prompt: str,
    schema: type[T],
    max_attempts: int = 3,
) -> T:
    """
    Invoke the LLM and validate its response against a Pydantic schema.

    Retries are limited to malformed or schema-invalid model outputs.
    Provider/API failures are propagated to the caller.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    instruction = f"""
You are a structured-output generator.

Return exactly ONE valid JSON object.
Do not include Markdown fences, explanations, or text outside the JSON.

Your response must conform to this JSON Schema:

{schema_json}

Original task:
{prompt}
"""

    current_prompt = instruction
    last_error = None

    for attempt in range(1, max_attempts + 1):
        response = llm.invoke(current_prompt)
        content = response.content

        if not isinstance(content, str):
            raise TypeError("Expected a text response from the LLM.")

        try:
            data = _extract_json_object(content)
            return schema.model_validate(data)

        except (ValueError, ValidationError) as exc:
            last_error = exc

            if attempt == max_attempts:
                break

            current_prompt = f"""
Your previous response was invalid.

Validation error:
{str(exc)}

Previous response:
{content[:3000]}

Return a corrected JSON object matching this schema exactly:

{schema_json}

Do not include Markdown fences or additional commentary.
"""

    raise ValueError(
        f"Could not obtain valid {schema.__name__} output "
        f"after {max_attempts} attempts. Last error: {last_error}"
    )
