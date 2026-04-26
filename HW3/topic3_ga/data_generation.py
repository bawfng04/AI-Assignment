from __future__ import annotations

import random

from .config import ProblemConfig
from .models import Offering, ProblemData, Room


def _build_course_professor_map(
    config: ProblemConfig,
    rng: random.Random,
) -> tuple[dict[int, int], dict[int, set[int]]]:
    professor_to_courses: dict[int, set[int]] = {
        professor_id: set() for professor_id in range(1, config.num_professors + 1)
    }
    course_to_professor: dict[int, int] = {}

    for course_id in range(1, config.num_courses + 1):
        eligible_professors = [
            professor_id
            for professor_id, courses in professor_to_courses.items()
            if len(courses) < config.max_courses_per_professor
        ]
        if not eligible_professors:
            eligible_professors = sorted(
                professor_to_courses,
                key=lambda professor_id: len(professor_to_courses[professor_id]),
            )
        professor_id = rng.choice(eligible_professors)
        professor_to_courses[professor_id].add(course_id)
        course_to_professor[course_id] = professor_id

    return course_to_professor, professor_to_courses


def _generate_rooms(config: ProblemConfig, rng: random.Random) -> list[Room]:
    rooms: list[Room] = []
    large_room_target = max(1, int(config.num_rooms * 0.4))
    large_rooms_created = 0

    for room_id in range(1, config.num_rooms + 1):
        if large_rooms_created < large_room_target:
            room_size = 1 if rng.random() < 0.65 else 0
        else:
            room_size = 0
        if room_size == 1:
            large_rooms_created += 1

        is_available = rng.random() > 0.08
        rooms.append(Room(room_id=room_id, room_size=room_size, is_available=is_available))

    if not any(room.is_available for room in rooms):
        first = rooms[0]
        rooms[0] = Room(room_id=first.room_id, room_size=first.room_size, is_available=True)

    return rooms


def _generate_offerings(
    config: ProblemConfig,
    course_to_professor: dict[int, int],
    rng: random.Random,
) -> list[Offering]:
    offerings: list[Offering] = []

    for offering_id in range(1, config.num_offerings + 1):
        course_id = rng.randint(1, config.num_courses)
        offerings.append(
            Offering(
                offering_id=offering_id,
                course_id=course_id,
                section_id=rng.randint(1, config.num_sections),
                professor_id=course_to_professor[course_id],
                class_registration_size=1 if rng.random() < 0.48 else 0,
            )
        )

    return offerings


def generate_problem_data(config: ProblemConfig) -> ProblemData:
    rng = random.Random(config.seed)
    course_to_professor, professor_to_courses = _build_course_professor_map(config, rng)
    rooms = _generate_rooms(config, rng)
    offerings = _generate_offerings(config, course_to_professor, rng)

    overload = sum(
        max(0, len(courses) - config.max_courses_per_professor)
        for courses in professor_to_courses.values()
    )

    return ProblemData(
        rooms=rooms,
        offerings=offerings,
        professor_to_courses=professor_to_courses,
        professor_course_overload_count=overload,
    )


def validate_problem_data(data: ProblemData) -> list[str]:
    warnings: list[str] = []

    size_one_classes = sum(
        1 for offering in data.offerings if offering.class_registration_size == 1
    )
    size_one_rooms = sum(
        1
        for room in data.rooms
        if room.room_size == 1 and room.is_available
    )
    if size_one_classes > 0 and size_one_rooms == 0:
        warnings.append(
            "No available large room for size-1 classes. Hard violations will dominate."
        )

    if data.professor_course_overload_count > 0:
        warnings.append(
            "Professor-to-course mapping exceeds max courses per professor."
        )

    available_rooms = sum(1 for room in data.rooms if room.is_available)
    if available_rooms < 3:
        warnings.append(
            "Very few rooms are available; schedule feasibility may be difficult."
        )

    return warnings
