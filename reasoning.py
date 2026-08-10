"""The step-by-step reasoning loop.

Yields plain dicts. The Flask layer turns them into server-sent events, and tests
consume them directly. Every path out of this generator ends in either a ``final``
or an ``error`` event, which is the property that fixes the reported symptom of a
page where nothing ever appears.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from backends import STEP_SCHEMA, BackendError, ChatBackend, EmbeddingBackend
from graph import build_graph, edge_weight_spread, serialize_graph_data, strongest_path

MAX_STEPS = 20
MIN_STEPS = 5
MAX_STEP_CHARS = 700
# How many times a single step may be re-asked before we take what we were given.
# The original code retried without bound and without incrementing the step
# counter, so a model that kept answering too long, or kept trying to finish early,
# held the loop forever while the browser sat waiting on a stream that never spoke.
MAX_RETRIES_PER_STEP = 3

SYSTEM_PROMPT = (
    "You are an expert AI assistant that explains your reasoning step by step. For each step, "
    "provide a title that describes what you're doing in that step, along with the content. "
    "Decide if you need another step or if you're ready to give the final answer. Respond in "
    "JSON format with 'title', 'content', and 'next_action' (either 'continue' or "
    "'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF "
    "YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE "
    "EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN "
    "YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. "
    "WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO "
    "SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. "
    "USE BEST PRACTICES. Keep the content of each step under "
    f"{MAX_STEP_CHARS} characters."
)


def extract_json(text: str) -> dict:
    """Best-effort parse of a step, for backends that ignore the response schema."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fall back to the last brace-delimited object in the text.
    for candidate in reversed(re.findall(r"\{[^{}]*\}", cleaned)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"title": "", "content": cleaned, "next_action": "continue"}


def _collect(chat: ChatBackend, messages: list[dict], max_tokens: int) -> str:
    return "".join(chat.stream(messages, max_tokens, schema=STEP_SCHEMA))


def _short_title(title: str, content: str, fallback: str) -> str:
    """A label short enough to read on a node, without spending an extra LLM call.

    The original made a second model call per step purely to shorten a title. That
    doubled the request count for something a slice does just as well.
    """
    text = (title or content or fallback).strip()
    text = " ".join(text.split())
    if len(text) <= 20:
        return text
    cut = text[:20]
    # Prefer a word boundary if one is close to the end.
    if " " in cut[10:]:
        cut = cut[: cut.rindex(" ")]
    return cut


def reason(
    prompt: str,
    *,
    chat: ChatBackend,
    embedder: EmbeddingBackend,
    store=None,
    max_steps: int = MAX_STEPS,
    min_steps: int = MIN_STEPS,
    top_k: int = 2,
) -> Iterator[dict]:
    """Run the reasoning loop, yielding one event dict at a time."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    node_ids: list[str] = []
    labels: list[str] = []
    vectors: list = []
    total_thinking_time = 0.0
    final_answer: str | None = None
    step_events = 0

    def graph_payload() -> tuple[dict, dict | None]:
        graph = build_graph(node_ids, labels, vectors, top_k=top_k)
        serialized = serialize_graph_data(graph)
        path, weights, mean = strongest_path(serialized)
        path_data = (
            {"strongest_path": path, "path_weights": weights, "avg_similarity": mean}
            if path is not None
            else None
        )
        return serialized, path_data

    try:
        while len(node_ids) < max_steps:
            step_number = len(node_ids) + 1
            step_json = None
            truncated = False

            for attempt in range(MAX_RETRIES_PER_STEP):
                started = time.time()
                raw = _collect(chat, messages, 300)
                total_thinking_time += time.time() - started
                step_json = extract_json(raw)
                content = str(step_json.get("content", "")).strip()

                if len(content) <= MAX_STEP_CHARS:
                    break
                if attempt == MAX_RETRIES_PER_STEP - 1:
                    # Take what we have rather than asking again forever.
                    step_json["content"] = content[:MAX_STEP_CHARS].rstrip() + "..."
                    truncated = True
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last response was too long. Give the same step again, "
                        f"under {MAX_STEP_CHARS} characters.",
                    }
                )

            assert step_json is not None
            content = str(step_json.get("content", "")).strip()
            if not content:
                content = "The model returned an empty step."
            title = str(step_json.get("title", "")).strip()
            next_action = str(step_json.get("next_action", "continue")).strip()

            node_id = f"Step{step_number}"
            node_ids.append(node_id)
            labels.append(f"Step {step_number}: {_short_title(title, content, node_id)}")
            vectors.append(embedder.embed([content])[0])
            if store is not None:
                store.add(content, vectors[-1], model=embedder.describe().get("model", ""))

            messages.append({"role": "assistant", "content": json.dumps(step_json)})
            wants_to_finish = next_action == "final_answer" or "boxed" in content.lower()

            if wants_to_finish and len(node_ids) >= min_steps:
                # This step *is* the answer. Announcing it as a step and then again as
                # the final answer would print the same text twice on the page and
                # draw two nodes over identical content.
                final_answer = content
                break

            serialized, path_data = graph_payload()
            step_events += 1
            yield {
                "type": "step",
                "step": step_number,
                "title": title or f"Step {step_number}",
                "content": content,
                "truncated": truncated,
                "graph": serialized,
                "path_data": path_data,
            }

            if wants_to_finish:
                # Nudge once per remaining step, but never without having made
                # progress -- the step counter has already advanced by now.
                messages.append(
                    {
                        "role": "user",
                        "content": f"You have given {len(node_ids)} of {min_steps} steps. Look for "
                        "errors or alternatives in your answer, then continue your reasoning.",
                    }
                )

        if final_answer is not None and node_ids:
            # The final answer *is* the last step. Relabel that node instead of adding
            # a second one holding the same text: two nodes over identical content
            # produce a similarity of exactly 1.00 between them, which is the
            # duplicate pair visible in this project's own example screenshot.
            labels[-1] = f"Final Answer: {_short_title('', final_answer, 'final')}"
        else:
            if final_answer is None:
                messages.append(
                    {"role": "user", "content": "Please provide the final answer based on your reasoning above."}
                )
                started = time.time()
                raw = _collect(chat, messages, 300)
                total_thinking_time += time.time() - started
                final_answer = str(extract_json(raw).get("content", raw)).strip() or "No final answer."

            node_ids.append(f"Step{len(node_ids) + 1}")
            labels.append(f"Final Answer: {_short_title('', final_answer, 'final')}")
            vectors.append(embedder.embed([final_answer])[0])
            if store is not None:
                store.add(final_answer, vectors[-1], model=embedder.describe().get("model", ""))

        serialized, path_data = graph_payload()
        yield {
            "type": "final",
            "content": final_answer,
            "graph": serialized,
            "path_data": path_data,
        }
        yield {
            "type": "done",
            "total_time": total_thinking_time,
            "steps": step_events,
            "edge_spread": edge_weight_spread(serialized),
            "embedding": embedder.describe(),
            "chat": chat.describe(),
        }

    except BackendError as exc:
        yield {"type": "error", "message": str(exc), "hint": exc.hint}
    except Exception as exc:  # noqa: BLE001 - the stream must always say what happened
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}", "hint": ""}
