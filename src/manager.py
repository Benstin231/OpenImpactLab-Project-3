import json
from dataclasses import dataclass
from google.genai import types
from .gemini_client import generate
from .researcher import ResearchResult


DELEGATE_PROMPT = """You are a Research Manager. Convert the sub-question into one precise, \
optimized search query for a Researcher Agent.
Output ONLY the query string (5-15 words), nothing else. No quotes, no punctuation at the end."""

EVALUATE_PROMPT = """You are a Research Quality Manager. Evaluate whether a Researcher Agent's \
response fully and accurately answers the original sub-question.

You will receive:
- The original sub-question
- The researcher's answer text
- The web search queries the researcher executed
- The source URLs returned by the search grounding system
- The iteration number

IMPORTANT CONTEXT: Sources are retrieved via Google Search Grounding and may appear as redirect \
or tracking URLs. Do NOT penalise for URL format. Instead, judge citation quality by whether \
the answer content is clearly drawn from real web sources and whether the search queries used \
were relevant and specific. A response backed by multiple targeted searches and grounded content \
should receive a high citation quality score even if the raw URLs look like redirects.

Evaluate against four criteria:
1. ACCURACY (1-5): Are the factual claims correct and free of hallucinations?
   - Score 1-2: Contains fabricated events, future awards presented as confirmed facts, or \
fictional projects treated as real.
   - Score 3: Mostly accurate but includes some unverifiable or speculative claims.
   - Score 4-5: All major claims are verifiable; no hallucinations detected.

2. COMPLETENESS (1-5): Does the answer cover all key aspects of the sub-question?
   - Score 1-2: Only addresses part of the question.
   - Score 3: Covers the main point but misses important sub-topics.
   - Score 4-5: Addresses all key dimensions of the question thoroughly.

3. RELEVANCE (1-5): Is the answer tightly focused on what was asked?
   - Score 1-2: Off-topic or heavily padded with irrelevant content.
   - Score 3: Mostly on-topic but drifts in places.
   - Score 4-5: Every paragraph directly answers the question.

4. CITATION_QUALITY (1-5): Is the answer clearly grounded in real web sources?
   - Score 1-2: No sources provided, or zero search queries were used.
   - Score 3: At least one search was run but few sources returned.
   - Score 4-5: Multiple targeted searches were run and sources are present. URL format \
(redirect vs direct) does NOT affect this score.

Respond in JSON only with this exact schema:
{
  "approved": <true or false>,
  "scores": {
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "relevance": <1-5>,
    "citation_quality": <1-5>
  },
  "critique": "<1-3 sentence explanation focusing on content quality, NOT URL format>",
  "follow_up_query": "<a concise 5-15 word search query to address specific content gaps, or null if approved>"
}

Approve (true) if ALL scores are >= 4.
If this is iteration 3, you MUST approve regardless of scores. Set approved to true and \
follow_up_query to null.
IMPORTANT: follow_up_query must address a specific CONTENT gap (missing facts, time period, \
topic area). It must be 5-15 words and be a search engine query, NOT a sentence or instruction. \
Never repeat a follow_up_query that was already used in a previous iteration."""


@dataclass
class EvaluationResult:
    approved: bool
    scores: dict
    critique: str
    follow_up_query: str | None


def delegate(sub_question: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=DELEGATE_PROMPT,
    )
    response = generate(sub_question, config)
    return (response.text or sub_question).strip()


def evaluate(sub_question: str, result: ResearchResult, iteration: int) -> EvaluationResult:
    sources_text = "\n".join(result.sources) if result.sources else "None"
    queries_text = "\n".join(result.search_queries_used) if result.search_queries_used else "None"

    user_message = f"""Original sub-question: {sub_question}

Web search queries executed by Researcher:
{queries_text}

Source URLs returned by search grounding:
{sources_text}

Researcher's answer:
{result.answer}

Iteration number: {iteration}"""

    config = types.GenerateContentConfig(
        system_instruction=EVALUATE_PROMPT,
        response_mime_type="application/json",
    )
    response = generate(user_message, config)

    try:
        data = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        return EvaluationResult(
            approved=iteration >= 3,
            scores={},
            critique="Failed to parse evaluation response.",
            follow_up_query=None,
        )

    return EvaluationResult(
        approved=bool(data.get("approved", False)),
        scores=data.get("scores", {}),
        critique=data.get("critique", ""),
        follow_up_query=data.get("follow_up_query") or None,
    )
