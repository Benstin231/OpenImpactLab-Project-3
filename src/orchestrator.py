import queue
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from . import director, manager, researcher
from .researcher import ResearchResult

MAX_ITERATIONS = 3

_request_queues: dict[str, queue.Queue] = {}
_queues_lock = threading.Lock()


@dataclass
class LogEvent:
    event_type: str
    data: dict
    timestamp: str


def create_queue(request_id: str) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _queues_lock:
        _request_queues[request_id] = q
    return q


def get_queue(request_id: str) -> queue.Queue | None:
    with _queues_lock:
        return _request_queues.get(request_id)


def remove_queue(request_id: str) -> None:
    with _queues_lock:
        _request_queues.pop(request_id, None)


def _emit(q: queue.Queue, event_type: str, data: dict) -> None:
    event = LogEvent(
        event_type=event_type,
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    q.put(asdict(event))


def run(prompt: str, request_id: str, q: queue.Queue) -> None:
    try:
        # Step 1: Director plans sub-questions
        _emit(q, "planning", {"message": "Analyzing your question and creating a research plan..."})
        sub_questions = director.plan(prompt)
        _emit(q, "planning", {"sub_questions": sub_questions, "count": len(sub_questions)})

        approved_results: list[ResearchResult] = []

        # Step 2: For each sub-question, Manager delegates and Researcher investigates
        for idx, sub_q in enumerate(sub_questions, 1):
            _emit(q, "delegation", {
                "sub_question": sub_q,
                "index": idx,
                "total": len(sub_questions),
                "message": f"Manager delegating sub-question {idx}/{len(sub_questions)}",
            })

            search_query = manager.delegate(sub_q)
            _emit(q, "delegation", {"search_query": search_query})

            current_query = search_query
            final_result: ResearchResult | None = None

            for iteration in range(1, MAX_ITERATIONS + 1):
                _emit(q, "research_start", {
                    "query": current_query,
                    "iteration": iteration,
                    "sub_question_index": idx,
                })

                result = researcher.research(current_query)

                _emit(q, "research_result", {
                    "answer_preview": result.answer[:300] + "..." if len(result.answer) > 300 else result.answer,
                    "sources": result.sources,
                    "search_queries_used": result.search_queries_used,
                    "iteration": iteration,
                    "sub_question_index": idx,
                })

                evaluation = manager.evaluate(sub_q, result, iteration)

                _emit(q, "evaluation", {
                    "approved": evaluation.approved,
                    "scores": evaluation.scores,
                    "critique": evaluation.critique,
                    "iteration": iteration,
                    "sub_question_index": idx,
                })

                if evaluation.approved or iteration == MAX_ITERATIONS:
                    _emit(q, "approved", {
                        "sub_question": sub_q,
                        "final_iteration": iteration,
                        "sub_question_index": idx,
                    })
                    final_result = result
                    break
                else:
                    next_query = evaluation.follow_up_query or current_query
                    _emit(q, "rejected", {
                        "critique": evaluation.critique,
                        "follow_up_query": next_query,
                        "iteration": iteration,
                        "sub_question_index": idx,
                    })
                    current_query = next_query

            if final_result:
                approved_results.append(final_result)

        # Step 3: Director synthesizes all results
        _emit(q, "synthesis_start", {
            "message": "Director synthesizing all research into a final answer...",
        })
        final_answer = director.synthesize(prompt, approved_results)

        all_sources: list[str] = []
        seen = set()
        for r in approved_results:
            for s in r.sources:
                if s not in seen:
                    seen.add(s)
                    all_sources.append(s)

        _emit(q, "done", {
            "answer": final_answer,
            "sources": all_sources,
        })

    except Exception as e:
        _emit(q, "error", {"message": str(e)})
        _emit(q, "done", {"answer": "", "sources": []})
