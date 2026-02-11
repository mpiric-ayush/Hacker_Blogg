from __future__ import annotations

import glob
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple
from new1 import app
import pandas as pd
import streamlit as st
# Make sure this import points to your graph file
#from new1 import app# ← your LangGraph compiled app

try:
    from new1 import app
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str) -> bytes:
    from io import BytesIO
    import zipfile

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current.update(next(iter(payload.values())))
        else:
            current.update(payload)
    return current


def render_markdown(md: str):
    st.markdown(md, unsafe_allow_html=False)


# -----------------------------
# Past blogs
# -----------------------------
def list_past_blogs() -> list[Path]:
    return sorted(
        [p for p in Path(".").glob("*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ───────────────────────────────────────────────
#               Streamlit App
# ───────────────────────────────────────────────
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=140, placeholder="e.g. Self-Attention Mechanism in Transformers explained")

    as_of = st.date_input("As-of date", value=date.today())

    run_btn = st.button("🚀 Generate Blog", type="primary", width="stretch")

    st.divider()
    st.subheader("Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No .md blog files found in current directory.")
    else:
        options = []
        file_map = {}
        for p in past_files[:40]:
            try:
                md = read_md_file(p)
                title = extract_title_from_md(md, p.stem)
            except:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_map[label] = p

        selected_label = st.radio(
            "Load existing blog",
            options=options,
            index=0,
            label_visibility="collapsed",
        )

        if st.button("📂 Load selected", width="stretch"):
            if selected_label in file_map:
                p = file_map[selected_label]
                md_text = read_md_file(p)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "final": md_text,
                }
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, p.stem)
                st.success(f"Loaded: {p.name}")
                st.rerun()


# Prefill topic if we just loaded a blog
if "topic_prefill" in st.session_state:
    # We don't set value= directly because widgets are controlled
    # But we can show a hint
    st.sidebar.info(f"Prefill suggestion: {st.session_state['topic_prefill']}")

# ───────────────────────────────────────────────
# Main layout
# ───────────────────────────────────────────────
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🧾 Logs"]
)

logs: list[str] = []


def log(msg: str):
    logs.append(msg)


def _count_tasks(plan_obj: Any) -> int:
    """Return number of tasks whether plan is dict, Pydantic model, or object."""
    if not plan_obj:
        return 0
    if isinstance(plan_obj, dict):
        return len(plan_obj.get("tasks", []))
    if hasattr(plan_obj, "model_dump"):
        try:
            return len(plan_obj.model_dump().get("tasks", []))
        except Exception:
            pass
    if hasattr(plan_obj, "tasks"):
        try:
            return len(getattr(plan_obj, "tasks") or [])
        except Exception:
            pass
    return 0


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "final": "",
    }

    with st.status("Generating blog…", expanded=True) as status:
        progress = st.empty()

        current_state: Dict[str, Any] = {}
        last_node = None

        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict):
                    if len(payload) == 1 and isinstance(v := next(iter(payload.values())), dict):
                        node_name = next(iter(payload))
                        current_state.update(v)
                    else:
                        node_name = "unknown"
                        current_state.update(payload)

                if node_name and node_name != last_node:
                    status.write(f"➡️ **{node_name}**")
                    last_node = node_name

                # Show compact progress
                summary = {
                    "mode": current_state.get("mode", "—"),
                    "research": current_state.get("needs_research", False),
                    "queries": len(current_state.get("queries", [])),
                    "evidence": len(current_state.get("evidence", [])),
                    "tasks": _count_tasks(current_state.get("plan")),
                    "sections": len(current_state.get("sections", [])),
                }
                progress.json(summary)

                log(f"[{kind}] {json.dumps(payload, default=str)[:800]}…")

            elif kind == "final":
                out = payload
                st.session_state["last_out"] = out
                status.update(label="✅ Finished", state="complete", expanded=False)
                log("[final] received complete state")


# ───────────────────────────────────────────────
# Render last result
# ───────────────────────────────────────────────
out = st.session_state.get("last_out")

if out:
    # Plan tab
    with tab_plan:
        st.subheader("Blog Plan")
        plan = out.get("plan")
        if not plan:
            st.info("No plan was generated (possibly closed-book mode or error).")
        else:
            if hasattr(plan, "model_dump"):
                p = plan.model_dump()
            elif isinstance(plan, dict):
                p = plan
            else:
                p = {"blog_title": "Unknown", "tasks": []}

            st.write("**Title**:", p.get("blog_title", "—"))
            cols = st.columns(3)
            cols[0].write(f"**Audience**: {p.get('audience', '—')}")
            cols[1].write(f"**Tone**: {p.get('tone', '—')}")
            cols[2].write(f"**Kind**: {p.get('blog_kind', 'explainer')}")

            tasks = p.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "research": t.get("requires_research"),
                            "citations": t.get("requires_citations"),
                            "code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags", [])),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, width="stretch", hide_index=True)

                with st.expander("Full task JSON"):
                    st.json(tasks)

    # Evidence tab
    with tab_evidence:
        st.subheader("Evidence / Sources")
        evidence = out.get("evidence", [])
        if not evidence:
            st.info("No evidence collected (likely closed_book mode or no search results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title", "—"),
                        "date": e.get("published_at", "—"),
                        "source": e.get("source", "—"),
                        "url": e.get("url", "—"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Preview tab
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final", "")
        if not final_md:
            st.warning("No final markdown content was generated.")
        else:
            render_markdown(final_md)

            title = "blog"
            plan = out.get("plan")
            if plan:
                if hasattr(plan, "blog_title"):
                    title = plan.blog_title
                elif isinstance(plan, dict):
                    title = plan.get("blog_title", "blog")

            # fallback from content
            if title == "blog":
                title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(title)}.md"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    width="stretch",
                )
            with col2:
                bundle = bundle_zip(final_md, md_filename)
                st.download_button(
                    "📦 Download .zip (MD only)",
                    data=bundle,
                    file_name=f"{safe_slug(title)}_bundle.zip",
                    mime="application/zip",
                    width="stretch",
                )

    # Logs tab
    with tab_logs:
        st.subheader("Execution Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area(
            "Recent events",
            value="\n\n".join(st.session_state["logs"][-100:]),
            height=500,
        )

else:
    st.info("Enter a topic and press **Generate Blog** to start.")
