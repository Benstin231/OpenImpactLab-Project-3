import json
from google.genai import types
from .gemini_client import generate
from .researcher import ResearchResult


PLAN_PROMPT = """You are a Research Director. Analyze the user's question and break it into \
2-4 focused sub-questions that, when answered together, will fully address the original question.

Output ONLY a JSON array of strings — no explanation, no markdown:
["sub-question 1", "sub-question 2", ...]"""

SYNTHESIZE_PROMPT = """You are a Research Director. You have received approved research results \
for each sub-question you delegated. Synthesize them into one coherent, well-structured final \
answer for the user.

Rules:
- Do not add new claims beyond the provided research.
- Cite sources inline using [Source: url] notation where relevant.
- Write in clear paragraphs, 3-6 total.
- Begin directly with the answer — no preamble like "Based on the research..."."""


def plan(prompt: str) -> list[str]:
    config = types.GenerateContentConfig(
        system_instruction=PLAN_PROMPT,
        response_mime_type="application/json",
    )
    response = generate(prompt, config)
    try:
        sub_questions = json.loads(response.text or "[]")
    except json.JSONDecodeError:
        return [prompt]
    if not isinstance(sub_questions, list) or not sub_questions:
        return [prompt]
    return [str(q) for q in sub_questions]


def synthesize(original_prompt: str, results: list[ResearchResult]) -> str:
    research_text = ""
    for i, result in enumerate(results, 1):
        sources_text = "\n".join(f"  - {s}" for s in result.sources) if result.sources else "  - None"
        research_text += f"\n--- Research Result {i} ---\n{result.answer}\nSources:\n{sources_text}\n"

    user_message = f"Original question: {original_prompt}\n\nResearch findings:{research_text}"

    config = types.GenerateContentConfig(
        system_instruction=SYNTHESIZE_PROMPT,
    )
    response = generate(user_message, config)
    return response.text or ""
