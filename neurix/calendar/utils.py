import json
from neurix.models import (CalendarTask, UserRoadmap)
from dataclasses import dataclass
from groq import Groq
from datetime import date, timedelta
from flask_login import current_user
from neurix import db
from flask import current_app


@dataclass
class CalCell:
    date: date
    in_month: bool


# ── helpers ──────────────────────────────────────────────────────────────────

def _seed_roadmap_tasks(user_roadmap: UserRoadmap) -> None:
    """Create CalendarTask rows for every RoadmapTask in the enrolment."""
    roadmap = user_roadmap.roadmap
    for rt in roadmap.tasks:
        task_date = user_roadmap.start_date + timedelta(days=rt.day_offset)
        # avoid duplicates if called twice
        exists = CalendarTask.query.filter_by(
            user_id=current_user.id,
            roadmap_task_id=rt.id,
        ).first()
        if not exists:
            ct = CalendarTask(
                user_id=current_user.id,
                date=task_date,
                title=rt.title,
                description=rt.description,
                source='roadmap',
                roadmap_task_id=rt.id,
            )
            db.session.add(ct)
    user_roadmap.tasks_seeded = True
    db.session.commit()


def _groq_generate_roadmap(topic: str, days: int) -> dict:
    """
    Call Groq (llama-3 70b) to produce a JSON roadmap.
    Returns a dict: {title, description, tasks: [{day_offset, title, description}]}
    Raises ValueError on bad JSON / schema mismatch.
    """
    client = Groq(api_key=current_app.config['GROQ_API_KEY'])

    system_prompt = (
        "You are a curriculum designer. When asked, output ONLY valid JSON "
        "with this exact schema (no markdown, no prose):\n"
        "{\n"
        '  "title": "<roadmap title>",\n'
        '  "description": "<one-sentence overview>",\n'
        '  "tasks": [\n'
        '    {"day_offset": <int 0-based>, "title": "<task>", "description": "<2-3 sentences>"}\n'
        "  ]\n"
        "}\n"
        "day_offset must start at 0 and increase monotonically. "
        "Spread tasks sensibly across the requested number of days."
    )

    user_prompt = (
        f"Create a {days}-day learning roadmap for the topic: {topic}. "
        f"Include one or two tasks per day. Keep titles concise (< 80 chars)."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()

    # strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    # basic schema validation
    required_keys = {"title", "description", "tasks"}
    if not required_keys.issubset(data):
        raise ValueError(f"Missing keys in AI response: {required_keys - set(data)}")
    for t in data["tasks"]:
        if not {"day_offset", "title", "description"}.issubset(t):
            raise ValueError("Malformed task in AI response")

    return data

