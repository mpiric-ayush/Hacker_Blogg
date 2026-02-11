from __future__ import annotations

import operator
import os
import re
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, List, Optional, Literal, Annotated, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ── Both LLM providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ───────────────────────────────────────────────
#          TWO MODELS — Gemini plans, Groq writes
# ───────────────────────────────────────────────
# Primary instance – used only for attribute look-ups (temperature /
# max_output_tokens) in the retry helpers.  Actual per-call instances are
# created inside _try_models_structured / _try_models_raw so they can swap
# model names on 404.
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",          # ← valid stable model
    temperature=0.7
)

groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120-550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# -----------------------------
# 2) State
# -----------------------------
class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    final: str


# -----------------------------
# 3) Retry / fallback helpers
# -----------------------------
def _safe_invoke_runner(
    runner,
    messages,
    fallback: Callable[[Optional[Exception]], object],
    retries: int = 3,
    backoff_base: float = 2.0,
):
    """
    Invoke runner.invoke(messages) with retries on transient errors.
    On final failure call fallback(exception) and return its result.
    """
    for attempt in range(retries):
        try:
            return runner.invoke(messages)
        except ChatGoogleGenerativeAIError as e:
            if attempt < retries - 1:
                time.sleep(backoff_base ** attempt)
                continue
            return fallback(e)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff_base ** attempt)
                continue
            return fallback(e)


def _candidate_model_names() -> list[str]:
    """
    Allow override via GEMINI_MODEL env var (only if it looks like a valid gemini model),
    then append a safe, explicit list of fallback model names.
    """
    env = os.getenv("GEMINI_MODEL")
    cands: list[str] = []

    # Accept only sensible model names (must start with "gemini-")
    if env:
        if re.match(r"(?i)^gemini[-_0-9a-z.]+$", env):
            cands.append(env)
        else:
            warnings.warn(f"Ignoring invalid GEMINI_MODEL value: {env!r}", stacklevel=2)

    # Explicit tuple/list of valid models — do NOT use a single string here
    for m in ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"):
        if m not in cands:
            cands.append(m)

    return cands


def _try_models_structured(
    schema,
    messages,
    fallback: Callable[[Optional[Exception]], object],
    retries: int = 3,
):
    """
    Try with_structured_output(schema) across candidate models.
    Returns the first successful structured result; on persistent
    failures returns fallback(exception).
    """
    candidates = _candidate_model_names()
    last_exc: Optional[Exception] = None
    for mdl in candidates:
        try:
            inst = ChatGoogleGenerativeAI(
                model=mdl,
                temperature=getattr(gemini, "temperature", 0.7),
                max_output_tokens=getattr(gemini, "max_output_tokens", 8192),
            )
            runner = inst.with_structured_output(schema)
            return _safe_invoke_runner(runner, messages, fallback, retries=retries)
        except ChatGoogleGenerativeAIError as e:
            last_exc = e
            if "NOT_FOUND" in str(e) or "404" in str(e):
                continue          # model doesn't exist -> try next
            return fallback(e)    # other error -> give up
        except Exception as e:
            last_exc = e
            continue
    return fallback(last_exc)


def _try_models_raw(
    messages,
    fallback: Callable[[Optional[Exception]], object],
    retries: int = 3,
):
    """
    Plain (non-structured) invocation across candidate models.
    Returns an object with .content on success, or fallback(...) on failure.
    """
    candidates = _candidate_model_names()
    last_exc: Optional[Exception] = None
    for mdl in candidates:
        try:
            inst = ChatGoogleGenerativeAI(
                model=mdl,
                temperature=getattr(gemini, "temperature", 0.7),
                max_output_tokens=getattr(gemini, "max_output_tokens", 8192),
            )
            return _safe_invoke_runner(inst, messages, fallback, retries=retries)
        except ChatGoogleGenerativeAIError as e:
            last_exc = e
            if "NOT_FOUND" in str(e) or "404" in str(e):
                continue
            return fallback(e)
        except Exception as e:
            last_exc = e
            continue
    return fallback(last_exc)


# -----------------------------
# 4) Router - Gemini
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3-10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""


def router_node(state: State) -> dict:
    decider_fallback = lambda e: RouterDecision(
        needs_research=False,
        mode="closed_book",
        reason=f"Fallback: gemini unavailable ({str(e)[:200]})",
        queries=[],
    )
    decision = _try_models_structured(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        fallback=decider_fallback,
    )

    recency_days = 7 if decision.mode == "open_book" else 45 if decision.mode == "hybrid" else 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 5) Research - Tavily + Gemini extraction
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor_fallback = lambda e: EvidencePack(evidence=[])
    pack = _try_models_structured(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ],
        fallback=extractor_fallback,
    )

    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 6) Orchestrator - Gemini
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and SEO-focused blog creator.

You MUST follow this exact Article Format when creating the plan:

Article Format (STRICT ORDER):
1. Main Title
2. Introduction (must mention the primary keyword naturally)
3. Definition (clearly define the topic in simple language)
4. Main Content Sections (optional, only if needed)
5. Conclusion
6. FAQs (3–5 questions with short answers)

Rules:
- Each section MUST be a separate Task.
- Tasks must be ordered exactly as above.
- FAQs must be a SINGLE task containing 3–5 questions.
- Introduction must be 1–2 short paragraphs.
- Definition must be simple and beginner-friendly.
- Conclusion must summarize and restate importance.
- No extra sections outside this structure.

Requirements:
- 5–9 tasks total.
- Each task must include:
  - goal
  - 3–6 bullets

TOTAL WORD COUNT RULE:
- The total sum of all task.target_words MUST be approximately 2500 words (+/-5%).
- Each task MUST have an explicit target_words value.
- Distribute words logically by section importance.
- Do NOT assign more than 600 words to a single section.

Grounding rules:
- closed_book: evergreen content only.
- hybrid: up-to-date examples allowed with citations.
- open_book: news-style content only (no tutorials).

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    def _fallback_plan(e: Optional[Exception]) -> Plan:
        t = Task(
            id=1,
            title="Overview",
            goal="One-sentence summary and next steps.",
            bullets=["What this topic is", "Why it matters", "Key takeaway"],
            target_words=250,
        )
        return Plan(
            blog_title=state.get("topic") or "Untitled",
            audience="developers",
            tone="informative",
            blog_kind="news_roundup" if state.get("mode") == "open_book" else "explainer",
            constraints=[],
            tasks=[t],
        )

    plan = _try_models_structured(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state.get('mode', 'closed_book')}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\n"
                    f"Evidence:\n{[e.model_dump() for e in state.get('evidence', [])][:16]}"
                )
            ),
        ],
        fallback=_fallback_plan,
    )

    # 🔒 2500-word enforcement
    TOTAL_WORDS = 2500
    FIXED_BUDGETS = {
        "introduction": 200,
        "definition": 200,
        "conclusion": 200,
        "faq": 400,
    }

    fixed_total = 0
    main_tasks = []

    for task in plan.tasks:
        title = task.title.lower().strip()
        matched = False
        for key, words in FIXED_BUDGETS.items():
            if key in title:
                task.target_words = words
                fixed_total += words
                matched = True
                break
        if not matched:
            main_tasks.append(task)

    remaining_words = TOTAL_WORDS - fixed_total
    if remaining_words > 0 and main_tasks:
        per_task = remaining_words // len(main_tasks)
        for task in main_tasks:
            task.target_words = per_task

    if state.get("mode") == "open_book":
        plan.blog_kind = "news_roundup"

    return {"plan": plan}
# -----------------------------
# 7) Fanout
# -----------------------------
def fanout(state: State):
    """Emit a Send message to 'worker' for each Task in the Plan."""
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 8) Worker - Gemini (raw text)
# -----------------------------
WORKER_SYSTEM = """You are a senior SEO blog writer.

Write ONE section of a blog post in Markdown.

STRICT FORMAT RULES:
- Output must start with: ## <Section Title>
- Follow the section's role exactly:
  - Introduction: 1–2 short paragraphs, mention primary keyword naturally.
  - Definition: Simple explanation, beginner-friendly.
  - Conclusion: Short summary + importance restated.
  - FAQs: 3–5 questions using ### Question format with short answers.
- Cover ALL bullets in order.
- Stay within target word count (+/-15%).

FAQs FORMAT:
### Question 1
Short answer.

### Question 2
Short answer.

Scope Guard:
- No tutorials unless explicitly requested.
- No new facts unless supported by Evidence URLs (if required).

Citations:
- If requires_citations=true, include Markdown links.

Code:
- Include code ONLY if requires_code=true.

Word Count Rule:
- You MUST aim to match target_words closely.
- Being too short is a failure.
- If needed, expand explanations with examples or clarity (not fluff)
"""
def worker_node(payload: dict) -> dict:
    if payload["task"]["title"].lower() == "faqs":
        payload["task"]["bullets"] = [
            "Answer each question clearly",
            "Keep answers short and simple",
            "Do not introduce new topics",
        ]
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    def _fallback_section(e: Optional[Exception]) -> SimpleNamespace:
        content = f"## {task.title}\n\nNot available: writing model error. ({str(e)[:200]})"
        return SimpleNamespace(content=content)

    resp = _try_models_raw(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ],
        fallback=_fallback_section,
    )

    section_md = getattr(resp, "content", str(resp)).strip()
    return {"sections": [(task.id, section_md)]}

# -----------------------------
# 9) Reducer - merge sections -> final markdown (no images)
# -----------------------------
def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def merge_content(state: State) -> dict:
    """Order sections by task-id, prepend H1, persist .md, expose as final."""
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    # persist to current working directory
    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(merged_md, encoding="utf-8")

    return {"merged_md": merged_md, "final": merged_md}
# -----------------------------
# 10) Build graph
# -----------------------------
# Reducer subgraph: single node that merges + finalises
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", END)
reducer_subgraph = reducer_graph.compile()

# Main graph
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
