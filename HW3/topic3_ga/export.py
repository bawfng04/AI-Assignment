from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from .config import DAY_NAMES, ScheduleSummary
from .models import EvaluationResult, RunArtifacts, ScheduleRow


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def schedule_rows_to_dicts(rows: list[ScheduleRow]) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for row in rows:
        day_name = DAY_NAMES[row.day] if 0 <= row.day < len(DAY_NAMES) else f"D{row.day}"
        records.append(
            {
                "offering_id": row.offering_id,
                "course_id": row.course_id,
                "section_id": row.section_id,
                "professor_id": row.professor_id,
                "session_index": row.session_index + 1,
                "day": day_name,
                "timeslot": row.timeslot + 1,
                "room_id": row.room_id,
            }
        )
    return records


def format_schedule_table(rows: list[ScheduleRow]) -> str:
    records = schedule_rows_to_dicts(rows)
    headers = [
        "Day",
        "Slot",
        "Room",
        "Section",
        "Course",
        "Professor",
        "Offering",
        "Session",
    ]

    table_rows = [
        [
            str(record["day"]),
            str(record["timeslot"]),
            str(record["room_id"]),
            str(record["section_id"]),
            str(record["course_id"]),
            str(record["professor_id"]),
            str(record["offering_id"]),
            str(record["session_index"]),
        ]
        for record in records
    ]

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    separator = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    body_lines = [
        " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in table_rows
    ]

    return "\n".join([header_line, separator, *body_lines])


def export_schedule_json(
    output_dir: str,
    file_name: str,
    rows: list[ScheduleRow],
    summary: ScheduleSummary,
    evaluation: EvaluationResult,
) -> str:
    payload = {
        "summary": asdict(summary),
        "best_evaluation": {
            "fitness": evaluation.fitness,
            "total_penalty": evaluation.total_penalty,
            "soft_penalty": evaluation.soft_penalty,
            "hard_breakdown": asdict(evaluation.hard),
        },
        "schedule": schedule_rows_to_dicts(rows),
    }

    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return file_path


def export_schedule_csv(output_dir: str, file_name: str, rows: list[ScheduleRow]) -> str:
    records = schedule_rows_to_dicts(rows)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "offering_id",
                "course_id",
                "section_id",
                "professor_id",
                "session_index",
                "day",
                "timeslot",
                "room_id",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return file_path


def build_artifacts(
    output_dir: str,
    json_path: str,
    csv_path: str,
    plot_path: str,
) -> RunArtifacts:
    return RunArtifacts(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
        plot_path=plot_path,
    )
