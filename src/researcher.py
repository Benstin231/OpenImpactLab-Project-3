from dataclasses import dataclass, field
from google.genai import types
from .gemini_client import generate

SYSTEM_PROMPT = """You are a Researcher Agent. Your sole job is to find accurate, \
current information on the given query using web search.

Rules:
- Retrieve factual data; do not speculate or fill gaps with prior knowledge when search results are available.
- Synthesize information from multiple sources when possible.
- Always include the specific URLs of pages that informed your answer.
- Keep your response concise but complete: answer the query directly in 2-4 paragraphs, then list sources.
- Format sources as a clean list at the end, each on its own line starting with "Source: <url>".
- Do not include conversational filler — just findings and sources."""


@dataclass
class ResearchResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)


def research(query: str) -> ResearchResult:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    response = generate(query, config)

    answer = response.text or ""
    sources: list[str] = []
    queries_used: list[str] = []

    if response.candidates:
        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            if grounding.web_search_queries:
                queries_used = list(grounding.web_search_queries)
            if grounding.grounding_chunks:
                seen = set()
                for chunk in grounding.grounding_chunks:
                    web = getattr(chunk, "web", None)
                    if web and web.uri and web.uri not in seen:
                        seen.add(web.uri)
                        sources.append(web.uri)

    return ResearchResult(answer=answer, sources=sources, search_queries_used=queries_used)
