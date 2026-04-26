from __future__ import annotations

from dataclasses import asdict
import io
import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from topic3_ga.config import DAY_NAMES, GAConfig, ProblemConfig, RunConfig
from topic3_ga.export import schedule_rows_to_dicts
from topic3_ga.main import run_scheduler
from topic3_ga.models import GenerationMetrics, ProblemData, ScheduleRow
from topic3_ga.multi_seed import parse_seeds, run_multi_seed_report


def _build_progress_figure(metrics: list[GenerationMetrics]) -> plt.Figure:
    generations = [item.generation for item in metrics]
    best_fitness = [item.best_fitness for item in metrics]
    avg_fitness = [item.avg_fitness for item in metrics]
    hard_violations = [item.best_hard_violations for item in metrics]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(generations, best_fitness, label="Best Fitness", color="#0B7285", linewidth=2.1)
    ax1.plot(generations, avg_fitness, label="Average Fitness", color="#D9480F", linewidth=1.9)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        generations,
        hard_violations,
        label="Best Hard Violations",
        color="#5F3DC4",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    ax2.set_ylabel("Hard Violations")

    lines_a, labels_a = ax1.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax1.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")
    fig.tight_layout()
    return fig


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _build_zip_bytes(paths: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=os.path.basename(path))
    return buffer.getvalue()


def _label(prefix: str, value: int, width: int = 2) -> str:
    return f"{prefix}{int(value):0{width}d}"


def _build_rooms_input_df(data: ProblemData) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for room in sorted(data.rooms, key=lambda item: item.room_id):
        records.append(
            {
                "Room": _label("R", room.room_id),
                "Room id": room.room_id,
                "Room size": room.room_size,
                "Available": "Yes" if room.is_available else "No",
            }
        )
    return pd.DataFrame(records)


def _build_offerings_input_df(data: ProblemData) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for offering in sorted(data.offerings, key=lambda item: item.offering_id):
        records.append(
            {
                "Offering": _label("O", offering.offering_id),
                "Course": _label("C", offering.course_id),
                "Section": _label("SEC", offering.section_id),
                "Professor": _label("P", offering.professor_id),
                "Class size": offering.class_registration_size,
            }
        )
    return pd.DataFrame(records)


def _build_professor_course_df(data: ProblemData) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for professor_id in sorted(data.professor_to_courses):
        courses = sorted(data.professor_to_courses[professor_id])
        records.append(
            {
                "Professor": _label("P", professor_id),
                "Courses count": len(courses),
                "Courses": ", ".join(_label("C", course_id) for course_id in courses),
            }
        )
    return pd.DataFrame(records)


def _build_readable_schedule_df(rows: list[ScheduleRow], data: ProblemData) -> pd.DataFrame:
    room_by_id = {room.room_id: room for room in data.rooms}
    records: list[dict[str, object]] = []
    for row in rows:
        room = room_by_id.get(row.room_id)
        day_name = DAY_NAMES[row.day] if 0 <= row.day < len(DAY_NAMES) else f"D{row.day}"
        records.append(
            {
                "Day": day_name,
                "Timeslot": row.timeslot + 1,
                "Room": _label("R", row.room_id),
                "Room size": room.room_size if room is not None else "?",
                "Section": _label("SEC", row.section_id),
                "Course": _label("C", row.course_id),
                "Professor": _label("P", row.professor_id),
                "Offering": _label("O", row.offering_id),
                "Session": row.session_index + 1,
            }
        )

    schedule_df = pd.DataFrame(records)
    if schedule_df.empty:
        return schedule_df

    day_order = {day_name: idx for idx, day_name in enumerate(DAY_NAMES)}
    schedule_df["_day_idx"] = schedule_df["Day"].map(day_order).fillna(999)
    schedule_df = schedule_df.sort_values(["_day_idx", "Timeslot", "Room", "Section", "Course"])
    return schedule_df.drop(columns=["_day_idx"]).reset_index(drop=True)


def _build_timetable_df(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame()

    work = schedule_df.copy()
    work["Entry"] = work.apply(
        lambda row: (
            f"{row['Course']} | {row['Section']} | {row['Professor']} | "
            f"{row['Room']} (S{int(row['Session'])})"
        ),
        axis=1,
    )

    grouped = (
        work.groupby(["Timeslot", "Day"], sort=True)["Entry"]
        .apply(lambda entries: "\n".join(entries))
        .reset_index()
    )
    pivot = grouped.pivot(index="Timeslot", columns="Day", values="Entry")
    pivot = pivot.reindex(columns=[day_name for day_name in DAY_NAMES if day_name in pivot.columns])
    pivot = pivot.fillna("-")
    pivot.index = [f"Slot {slot}" for slot in pivot.index]
    pivot = pivot.reset_index().rename(columns={"index": "Timeslot"})
    return pivot


def _csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def _single_run_tab() -> None:
    st.subheader("Single Run")
    st.caption("Quick input, run GA, watch generation metrics, inspect schedule, and download artifacts.")

    with st.form("single_run_form"):
        c1, c2, c3 = st.columns(3)
        seed = c1.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
        offerings = c2.number_input("Offerings", min_value=10, max_value=80, value=30, step=1)
        population = c3.number_input("Population", min_value=20, max_value=1000, value=240, step=10)

        c4, c5, c6 = st.columns(3)
        generations = c4.number_input("Generations", min_value=10, max_value=2000, value=420, step=10)
        mutation_rate = c5.number_input("Mutation rate", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        crossover_rate = c6.number_input("Crossover rate", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

        c7, c8, c9 = st.columns(3)
        elitism = c7.number_input("Elitism", min_value=1, max_value=200, value=10, step=1)
        tournament_size = c8.number_input("Tournament size", min_value=2, max_value=20, value=4, step=1)
        no_improvement_patience = c9.number_input(
            "No improvement patience",
            min_value=1,
            max_value=2000,
            value=100,
            step=1,
        )

        c10, c11, c12 = st.columns(3)
        feasible_streak_patience = c10.number_input(
            "Feasible streak patience",
            min_value=1,
            max_value=2000,
            value=100,
            step=1,
        )
        run_name = c11.text_input("Run name", value="ui_demo")
        output_dir = c12.text_input("Output directory", value="outputs")

        run_button = st.form_submit_button("Run GA and export artifacts", use_container_width=True)

    if not run_button:
        return

    progress = st.progress(0.0)
    status = st.empty()
    chart_placeholder = st.empty()

    generation_snapshots: list[GenerationMetrics] = []

    def on_generation(metric: GenerationMetrics) -> None:
        generation_snapshots.append(metric)
        if len(generation_snapshots) == 1 or len(generation_snapshots) % 5 == 0:
            fig = _build_progress_figure(generation_snapshots)
            chart_placeholder.pyplot(fig, use_container_width=True)
            plt.close(fig)
        progress.progress(min((metric.generation + 1) / float(generations), 1.0))
        status.caption(
            "Generation {} | best fitness {:.8f} | avg fitness {:.8f} | hard violations {}".format(
                metric.generation,
                metric.best_fitness,
                metric.avg_fitness,
                metric.best_hard_violations,
            )
        )

    problem_config = ProblemConfig(seed=int(seed), num_offerings=int(offerings))
    ga_config = GAConfig(
        population_size=int(population),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        elitism_count=int(elitism),
        tournament_size=int(tournament_size),
        no_improvement_patience=int(no_improvement_patience),
        feasible_streak_patience=int(feasible_streak_patience),
    )
    run_config = RunConfig(
        output_dir=output_dir,
        run_name=run_name,
        print_schedule_table=False,
    )

    with st.spinner("Running genetic algorithm..."):
        data, result, rows, artifacts = run_scheduler(
            problem_config,
            ga_config,
            run_config,
            generation_callback=on_generation,
        )

    if generation_snapshots:
        fig = _build_progress_figure(generation_snapshots)
        chart_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.success("Run completed.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best fitness", f"{result.summary.best_fitness:.8f}")
    m2.metric("Best hard violations", str(result.summary.best_hard_violations))
    m3.metric("Best soft penalty", f"{result.summary.best_soft_penalty:.4f}")
    m4.metric("Generations run", str(result.summary.generations_run))

    with st.expander("Input parameters used for this run", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.markdown("**Problem config**")
        p1.json(asdict(problem_config), expanded=False)
        p2.markdown("**GA config**")
        p2.json(asdict(ga_config), expanded=False)
        p3.markdown("**Run config**")
        p3.json(asdict(run_config), expanded=False)

    st.markdown("#### Generated input data snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rooms", str(len(data.rooms)))
    c2.metric("Available rooms", str(sum(1 for room in data.rooms if room.is_available)))
    c3.metric("Offerings", str(len(data.offerings)))
    c4.metric(
        "Professor overload",
        str(data.professor_course_overload_count),
    )

    with st.expander("Input offerings (generated by seed)", expanded=False):
        st.dataframe(_build_offerings_input_df(data), use_container_width=True, height=320)

    with st.expander("Input rooms", expanded=False):
        st.dataframe(_build_rooms_input_df(data), use_container_width=True, height=300)

    with st.expander("Professor to courses mapping", expanded=False):
        st.dataframe(_build_professor_course_df(data), use_container_width=True, height=260)

    st.markdown("#### Hard-constraint breakdown")
    st.json(asdict(result.best_evaluation.hard), expanded=False)

    st.markdown("#### Final schedule (readable)")
    schedule_df = _build_readable_schedule_df(rows, data)
    st.dataframe(schedule_df, use_container_width=True, height=420)

    st.markdown("#### Weekly timetable view (for report/demo)")
    timetable_df = _build_timetable_df(schedule_df)
    st.dataframe(timetable_df, use_container_width=True, height=320)

    st.markdown("#### Raw schedule table (ID-based)")
    raw_table = pd.DataFrame(schedule_rows_to_dicts(rows))
    st.dataframe(raw_table, use_container_width=True, height=280)

    st.markdown("#### Saved fitness plot")
    st.image(artifacts.plot_path, caption=artifacts.plot_path)

    single_artifact_paths = [artifacts.json_path, artifacts.csv_path, artifacts.plot_path]
    single_zip = _build_zip_bytes(single_artifact_paths)
    st.download_button(
        label="Download single-run artifacts (zip)",
        data=single_zip,
        file_name=f"{run_name}_seed{seed}_artifacts.zip",
        mime="application/zip",
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        label="Download JSON",
        data=_read_bytes(artifacts.json_path),
        file_name=os.path.basename(artifacts.json_path),
        mime="application/json",
        use_container_width=True,
    )
    c2.download_button(
        label="Download CSV",
        data=_read_bytes(artifacts.csv_path),
        file_name=os.path.basename(artifacts.csv_path),
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        label="Download PNG",
        data=_read_bytes(artifacts.plot_path),
        file_name=os.path.basename(artifacts.plot_path),
        mime="image/png",
        use_container_width=True,
    )

    st.download_button(
        label="Download readable schedule CSV (for report)",
        data=_csv_bytes(schedule_df),
        file_name=f"{run_name}_seed{seed}_readable_schedule.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _multi_seed_tab() -> None:
    st.subheader("Multi-seed Summary")
    st.caption("Run multiple seeds and export summary CSV/Markdown/JSON for report.")

    with st.form("multi_seed_form"):
        c1, c2, c3 = st.columns(3)
        seeds_raw = c1.text_input("Seeds (comma-separated)", value="40,41,42,43,44")
        offerings = c2.number_input("Offerings", min_value=10, max_value=80, value=30, step=1)
        population = c3.number_input("Population", min_value=20, max_value=1000, value=240, step=10)

        c4, c5, c6 = st.columns(3)
        generations = c4.number_input("Generations", min_value=10, max_value=2000, value=420, step=10)
        mutation_rate = c5.number_input("Mutation rate", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        crossover_rate = c6.number_input("Crossover rate", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

        c7, c8, c9 = st.columns(3)
        elitism = c7.number_input("Elitism", min_value=1, max_value=200, value=10, step=1)
        tournament_size = c8.number_input("Tournament size", min_value=2, max_value=20, value=4, step=1)
        no_improvement_patience = c9.number_input(
            "No improvement patience",
            min_value=1,
            max_value=2000,
            value=100,
            step=1,
            key="multi_no_improvement",
        )

        c10, c11, c12 = st.columns(3)
        feasible_streak_patience = c10.number_input(
            "Feasible streak patience",
            min_value=1,
            max_value=2000,
            value=100,
            step=1,
            key="multi_feasible_streak",
        )
        run_name = c11.text_input("Run name", value="ui_multiseed")
        output_dir = c12.text_input("Output directory", value="outputs")

        run_button = st.form_submit_button("Run multi-seed and export summary", use_container_width=True)

    if not run_button:
        return

    try:
        seeds = parse_seeds(seeds_raw)
    except ValueError as error:
        st.error(str(error))
        return

    ga_config = GAConfig(
        population_size=int(population),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        elitism_count=int(elitism),
        tournament_size=int(tournament_size),
        no_improvement_patience=int(no_improvement_patience),
        feasible_streak_patience=int(feasible_streak_patience),
    )

    with st.spinner("Running multi-seed evaluation..."):
        rows, stats, paths = run_multi_seed_report(
            seeds=seeds,
            offerings=int(offerings),
            ga_config=ga_config,
            output_dir=output_dir,
            run_name=run_name,
        )

    st.success("Multi-seed summary completed.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs", str(int(stats["runs"])))
    m2.metric("Feasible runs", str(int(stats["feasible_runs"])))
    m3.metric("Feasible rate", f"{stats['feasible_rate']:.2%}")
    m4.metric("Avg soft penalty", f"{stats['avg_soft']:.4f}")

    st.markdown("#### Per-seed summary table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

    st.markdown("#### Summary markdown preview")
    with open(paths["summary_md"], "r", encoding="utf-8") as handle:
        st.markdown(handle.read())

    summary_paths = [paths["summary_csv"], paths["summary_md"], paths["summary_json"]]
    summary_zip = _build_zip_bytes(summary_paths)
    st.download_button(
        label="Download summary artifacts (zip)",
        data=summary_zip,
        file_name=f"{run_name}_summary_artifacts.zip",
        mime="application/zip",
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        label="Download summary CSV",
        data=_read_bytes(paths["summary_csv"]),
        file_name=os.path.basename(paths["summary_csv"]),
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        label="Download summary MD",
        data=_read_bytes(paths["summary_md"]),
        file_name=os.path.basename(paths["summary_md"]),
        mime="text/markdown",
        use_container_width=True,
    )
    c3.download_button(
        label="Download summary JSON",
        data=_read_bytes(paths["summary_json"]),
        file_name=os.path.basename(paths["summary_json"]),
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="HW3 Topic 3 GA Demo",
        layout="wide",
    )
    st.title("HW3 Topic 3 - Class Scheduling GA")
    st.caption(
        "Interactive demo UI for parameter input, generation tracking, schedule visualization, and artifact export."
    )

    tab_single, tab_multi = st.tabs(["Single run", "Multi-seed report"])

    with tab_single:
        _single_run_tab()

    with tab_multi:
        _multi_seed_tab()


if __name__ == "__main__":
    main()
