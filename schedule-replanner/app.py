"""
Streamlit dashboard for CPM delay-impact analysis.

All dates come from scheduler.py. The optional assistant in agent.py is used
only on the AI tab and must not block the rest of the app.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import scheduler as sch

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "activities.csv"

st.set_page_config(
    page_title="Godrej Activity Replanner",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_baseline() -> tuple[pd.DataFrame, sch.ProjectSchedule]:
    return sch.load_and_schedule(DATA_PATH)


def init_state() -> None:
    if "baseline_loaded" not in st.session_state:
        frame, schedule = load_baseline()
        st.session_state.baseline_loaded = True
        st.session_state.activity_frame = frame
        st.session_state.baseline = schedule
        st.session_state.impact = None
        st.session_state.revised_schedule = None
        st.session_state.analysis_error = None
        st.session_state.chat_history = []
        st.session_state.chat_log = []


def style_app() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; }
        .hint { color: #5c6778; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def schedule_table(schedule: sch.ProjectSchedule) -> pd.DataFrame:
    table = schedule.to_dataframe()
    table["critical"] = table["is_critical"].map({True: "Yes", False: "No"})
    return table


def build_gantt(schedule: sch.ProjectSchedule, tower_filter: str) -> go.Figure:
    rows = []
    for item in schedule.activities.values():
        if tower_filter != "All" and item.tower_id != tower_filter:
            continue
        rows.append(
            {
                "activity_id": item.activity_id,
                "label": f"{item.activity_id} · {item.activity_name}",
                "tower_id": item.tower_id,
                "start": item.earliest_start,
                "end": item.earliest_finish + timedelta(days=1),
                "critical": "Critical" if item.is_critical else "Non-critical",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title="No activities match this tower filter.")
        return fig
    frame = frame.sort_values(["start", "activity_id"])
    fig = px.timeline(
        frame,
        x_start="start",
        x_end="end",
        y="label",
        color="critical",
        color_discrete_map={"Critical": "#c0392b", "Non-critical": "#2980b9"},
        hover_data=["activity_id", "tower_id"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(420, 22 * len(frame)),
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        title="Gantt-style earliest-start timeline (bar includes the inclusive finish day)",
    )
    return fig


def build_network_figure(frame: pd.DataFrame, schedule: sch.ProjectSchedule) -> go.Figure:
    graph = sch.build_graph(frame)
    layers = list(nx.topological_generations(graph))
    positions: dict[str, tuple[float, float]] = {}
    for column, generation in enumerate(layers):
        ordered = sorted(generation)
        count = max(len(ordered) - 1, 1)
        for row_index, activity_id in enumerate(ordered):
            y = 0.5 if len(ordered) == 1 else 1 - row_index / count
            positions[activity_id] = (column, y)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    colors = []
    labels = []
    hover = []
    for activity_id, (x, y) in positions.items():
        item = schedule.activities[activity_id]
        node_x.append(x)
        node_y.append(y)
        colors.append("#c0392b" if item.is_critical else "#2980b9")
        labels.append(activity_id)
        hover.append(
            f"{activity_id} · {item.activity_name}<br>"
            f"Tower {item.tower_id}<br>"
            f"{item.earliest_start} → {item.earliest_finish}<br>"
            f"Float {item.total_float_days} day(s)"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#b0b8c1", width=1),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Finish-to-start dependency graph (red = critical)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#f7f9fc",
    )
    return fig


def run_delay_analysis(activity_id: str, delay_days: int) -> None:
    st.session_state.analysis_error = None
    try:
        _, _, revised, impact = sch.apply_delay(
            st.session_state.activity_frame,
            activity_id,
            delay_days,
        )
    except sch.ScheduleError as exc:
        st.session_state.impact = None
        st.session_state.revised_schedule = None
        st.session_state.analysis_error = str(exc)
        return
    st.session_state.revised_schedule = revised
    st.session_state.impact = impact


def render_overview(baseline: sch.ProjectSchedule) -> None:
    impact = st.session_state.impact
    delayed_count = impact["affected_activity_count"] if impact else 0
    finish_value = (
        impact["revised_project_finish"] if impact else baseline.project_finish.isoformat()
    )
    slip = impact["project_finish_slip_days"] if impact else 0

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Baseline completion", baseline.project_finish.isoformat())
    col_b.metric("Activities", len(baseline.activities))
    col_c.metric("Critical activities", len(baseline.critical_activities))
    col_d.metric("Affected after delay", delayed_count)

    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Displayed completion", finish_value)
    col_f.metric("Schedule variance (days)", slip)
    col_g.metric("Project duration (days)", baseline.project_duration_days)

    st.caption(
        "Demo dataset: typical multi-tower finish-to-start sequence with unique "
        "activity IDs. It is **not** a live Godrej Properties programme export. "
        "Durations are calendar days; weekends are not skipped. Version 1 has no "
        "resource leveling or recovery (crashing/overtime)."
    )
    st.write(
        "**Baseline critical path:** "
        + " → ".join(baseline.critical_path)
    )


def render_activity_table(schedule: sch.ProjectSchedule) -> None:
    table = schedule_table(schedule)
    towers = ["All", *sorted(table["tower_id"].unique())]
    col_search, col_tower, col_crit = st.columns([2, 1, 1])
    query = col_search.text_input("Search ID, name, or tower", value="")
    tower = col_tower.selectbox("Tower", towers)
    critical_only = col_crit.checkbox("Critical only", value=False)

    view = table.copy()
    if tower != "All":
        view = view[view["tower_id"] == tower]
    if critical_only:
        view = view[view["is_critical"] == True]  # noqa: E712
    if query.strip():
        needle = query.strip().casefold()
        mask = (
            view["activity_id"].str.casefold().str.contains(needle, regex=False)
            | view["activity_name"].str.casefold().str.contains(needle, regex=False)
            | view["tower_id"].str.casefold().str.contains(needle, regex=False)
        )
        view = view[mask]

    if view.empty:
        st.info("No activities match the current search and filters.")
        return

    display_cols = [
        "activity_id",
        "activity_name",
        "tower_id",
        "predecessors",
        "duration_days",
        "earliest_start",
        "earliest_finish",
        "total_float_days",
        "critical",
    ]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(view)} of {len(table)} activities.")


def render_impact() -> None:
    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)
        return
    impact = st.session_state.impact
    if impact is None:
        st.info(
            "No delay analysis yet. Choose an activity in the sidebar, set delay "
            "days, and click **Run delay-impact analysis**."
        )
        return

    left, right = st.columns(2)
    left.metric("Baseline completion", impact["original_project_finish"])
    right.metric(
        "Revised completion",
        impact["revised_project_finish"],
        delta=f"{impact['project_finish_slip_days']} day(s)",
    )
    if impact["project_finish_slip_days"] == 0:
        st.success(
            "Project completion did not move. Available total float absorbed the delay."
        )
    else:
        st.warning(
            f"Project completion slipped by {impact['project_finish_slip_days']} "
            "calendar day(s) because the delay could not be absorbed by float."
        )

    st.write(
        f"**Delayed activity:** `{impact['delayed_activity_id']}` "
        f"({impact['delayed_activity_name']}, tower {impact['delayed_tower_id']}) "
        f"by {impact['delay_days']} day(s)."
    )
    st.write("**Original critical path:** " + " → ".join(impact["original_critical_path"]))
    st.write("**Revised critical path:** " + " → ".join(impact["revised_critical_path"]))
    if impact["critical_path_changed"]:
        st.info("The critical path changed after the delay.")

    affected = pd.DataFrame(impact["affected_activities"])
    if affected.empty:
        st.info("No activity dates moved.")
        return
    st.subheader("Affected activities")
    show = affected[
        [
            "activity_id",
            "activity_name",
            "tower_id",
            "original_start",
            "revised_start",
            "start_slip_days",
            "original_finish",
            "revised_finish",
            "finish_slip_days",
            "original_float_days",
            "revised_float_days",
            "was_critical",
            "is_critical",
        ]
    ]
    st.dataframe(show, use_container_width=True, hide_index=True)


def csv_bytes_for_download() -> tuple[bytes, str]:
    if st.session_state.revised_schedule is not None:
        table = st.session_state.revised_schedule.to_dataframe()
        name = "revised_schedule.csv"
    else:
        table = st.session_state.baseline.to_dataframe()
        name = "baseline_schedule.csv"
    return table.to_csv(index=False).encode("utf-8"), name


def render_assistant() -> None:
    try:
        from agent import (
            SETUP_INSTRUCTIONS,
            AgentError,
            AgentUnavailableError,
            ScheduleAssistant,
            check_ollama_status,
        )
    except Exception as exc:  # pragma: no cover - import guard for the UI
        st.error(f"Could not load the optional assistant module: {exc}")
        return

    status = check_ollama_status()
    if not status["model_ready"]:
        st.warning("AI assistant is optional and currently unavailable.")
        st.markdown(SETUP_INSTRUCTIONS.replace("\n", "  \n"))
        if status.get("detail"):
            st.caption(f"Technical detail: {status['detail']}")
        st.info("Use the sidebar delay analysis to demonstrate the project without AI.")
        return

    st.success(status["message"])
    st.caption(
        "Answers are grounded only when a scheduler tool actually ran. "
        "The model cannot execute arbitrary code."
    )
    for item in st.session_state.chat_log:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item.get("tools"):
                names = ", ".join(tool["name"] for tool in item["tools"])
                st.caption(f"Tools executed: {names}")

    question = st.chat_input("Ask about completion date, critical path, or a delay")
    if not question:
        return
    st.session_state.chat_log.append({"role": "user", "content": question})
    with st.spinner("Consulting the schedule tools..."):
        try:
            assistant = ScheduleAssistant(DATA_PATH)
            result = assistant.ask(question, history=st.session_state.chat_history)
            answer = result["answer"]
            if not result["grounded"]:
                answer += (
                    "\n\n_Note: no scheduler tool returned a successful result "
                    "for this turn, so treat dates with caution._"
                )
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )
            st.session_state.chat_log.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "tools": result["tools_used"],
                }
            )
        except AgentUnavailableError as exc:
            st.session_state.chat_log.append(
                {"role": "assistant", "content": str(exc), "tools": []}
            )
        except AgentError as exc:
            st.session_state.chat_log.append(
                {"role": "assistant", "content": str(exc), "tools": []}
            )
    st.rerun()


def main() -> None:
    style_app()
    init_state()
    baseline: sch.ProjectSchedule = st.session_state.baseline
    frame: pd.DataFrame = st.session_state.activity_frame

    st.title("Godrej Activity Replanner")
    st.subheader("Construction delay-impact analysis with Critical Path Method")

    with st.sidebar:
        st.header("Delay scenario")
        st.caption("Pick a unique activity ID so Tower T1 tiling is never confused with T3.")
        activity_ids = sorted(frame["activity_id"].tolist())
        default_index = activity_ids.index("T3-TIL") if "T3-TIL" in activity_ids else 0
        activity_id = st.selectbox("Activity ID", activity_ids, index=default_index)
        selected = baseline.activities[activity_id]
        st.write(f"{selected.activity_name} · Tower {selected.tower_id}")
        st.write(
            f"Baseline: {selected.earliest_start} → {selected.earliest_finish} · "
            f"float {selected.total_float_days} day(s) · "
            f"{'critical' if selected.is_critical else 'non-critical'}"
        )
        delay_days = st.number_input(
            "Delay (calendar days)",
            min_value=0,
            max_value=365,
            value=5,
            step=1,
        )
        if st.button("Run delay-impact analysis", type="primary", use_container_width=True):
            with st.spinner("Recalculating the CPM network..."):
                run_delay_analysis(activity_id, int(delay_days))

        csv_bytes, csv_name = csv_bytes_for_download()
        st.download_button(
            "Download schedule CSV",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
            use_container_width=True,
        )
        if st.session_state.revised_schedule is None:
            st.caption("Download contains the baseline until you run an analysis.")
        else:
            st.caption("Download contains the revised schedule from the last analysis.")

        st.divider()
        st.caption(f"Data file: `{DATA_PATH.name}`")

    tab_overview, tab_table, tab_impact, tab_visuals, tab_ai = st.tabs(
        [
            "Overview",
            "Activity table",
            "Delay impact",
            "Timeline & network",
            "AI assistant",
        ]
    )
    with tab_overview:
        render_overview(baseline)
    with tab_table:
        current = st.session_state.revised_schedule or baseline
        if st.session_state.revised_schedule is not None:
            st.caption("Showing the revised schedule from the last delay analysis.")
        render_activity_table(current)
    with tab_impact:
        render_impact()
    with tab_visuals:
        current = st.session_state.revised_schedule or baseline
        towers = ["All", *sorted({item.tower_id for item in current.activities.values()})]
        tower_filter = st.selectbox("Gantt tower filter", towers, key="gantt_tower")
        st.plotly_chart(build_gantt(current, tower_filter), use_container_width=True)
        st.plotly_chart(build_network_figure(frame, current), use_container_width=True)
    with tab_ai:
        render_assistant()


if __name__ == "__main__":
    main()
