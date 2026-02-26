"""
neurix/calendar/routes.py
─────────────────────────
All calendar routes:
  GET  /calendar/                 – main calendar view
  GET  /calendar/tasks/<iso_date> – AJAX: tasks for one day (JSON)
  POST /calendar/tasks/add        – add a manual task
  POST /calendar/tasks/<id>/status – toggle done/pending/skipped (AJAX)
  POST /calendar/tasks/<id>/delete – delete a manual task
  GET  /calendar/roadmaps         – browse available roadmaps
  POST /calendar/roadmaps/generate – AI generates a new roadmap via Groq
  POST /calendar/roadmaps/<id>/enroll – enrol user in roadmap + seed tasks
  POST /calendar/roadmaps/<id>/unenroll – drop roadmap + remove future tasks
"""

import calendar as _cal
import json
from dataclasses import dataclass
from datetime import date, timedelta

from flask import (abort, current_app, flash, jsonify,
                   redirect, render_template, request, url_for)
from flask_login import current_user, login_required
from groq import Groq

from neurix import db
from neurix.calendar import calendar
from neurix.models import (CalendarTask, Roadmap, RoadmapTask, UserRoadmap)


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


# ── views ─────────────────────────────────────────────────────────────────────

@calendar.route('/')
@login_required
def index():
    today = date.today()
    year  = request.args.get('year',  today.year,  type=int)
    month = request.args.get('month', today.month, type=int)

    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    start = first_day - timedelta(days=first_day.weekday())
    end   = last_day  + timedelta(days=(6 - last_day.weekday()))
    calendar_cells = []
    cur = start
    while cur <= end:
        calendar_cells.append(CalCell(date=cur, in_month=(cur.month == month)))
        cur += timedelta(days=1)

    tasks_in_month = CalendarTask.query.filter(
        CalendarTask.user_id == current_user.id,
        CalendarTask.date    >= first_day,
        CalendarTask.date    <= last_day,
        ).all()

    day_map = {}
    for t in tasks_in_month:
        key = t.date.isoformat()
        if key not in day_map:
            day_map[key] = {'done': 0, 'pending': 0, 'skipped': 0, 'total': 0}
        day_map[key][t.status] += 1
        day_map[key]['total']  += 1

    active_roadmaps = (
        db.session.query(UserRoadmap, Roadmap)
        .join(Roadmap, UserRoadmap.roadmap_id == Roadmap.id)
        .filter(UserRoadmap.user_id == current_user.id)
        .all()
    )

    month_name = first_day.strftime('%B')

    return render_template(
        'calendar/index.html',
        title='My Calendar',
        today=today,
        year=year,
        month=month,
        month_name=month_name,
        first_day=first_day,
        last_day=last_day,
        prev_year=prev_year, prev_month=prev_month,
        next_year=next_year, next_month=next_month,
        day_map=json.dumps(day_map),
        calendar_cells=calendar_cells,
        active_roadmaps=active_roadmaps,
    )


@calendar.route('/tasks/<iso_date>')
@login_required
def tasks_for_day(iso_date: str):
    """AJAX endpoint – returns tasks for a given date as JSON."""
    try:
        target = date.fromisoformat(iso_date)
    except ValueError:
        abort(400)

    tasks = CalendarTask.query.filter_by(
        user_id=current_user.id,
        date=target,
    ).order_by(CalendarTask.source, CalendarTask.id).all()

    return jsonify([t.to_dict() for t in tasks])


@calendar.route('/tasks/add', methods=['POST'])
@login_required
def add_task():
    """Add a manual CalendarTask. Accepts JSON or form data."""
    data = request.get_json(silent=True) or request.form

    iso_date    = data.get('date', '')
    title       = (data.get('title') or '').strip()
    description = (data.get('description') or '').strip() or None

    if not iso_date or not title:
        if request.is_json:
            return jsonify({'error': 'date and title are required'}), 400
        flash('Date and title are required.', 'danger')
        return redirect(url_for('calendar.index'))

    try:
        task_date = date.fromisoformat(iso_date)
    except ValueError:
        if request.is_json:
            return jsonify({'error': 'invalid date format'}), 400
        flash('Invalid date.', 'danger')
        return redirect(url_for('calendar.index'))

    task = CalendarTask(
        user_id=current_user.id,
        date=task_date,
        title=title,
        description=description,
        source='manual',
    )
    db.session.add(task)
    db.session.commit()

    if request.is_json:
        return jsonify(task.to_dict()), 201

    flash('Task added!', 'success')
    return redirect(url_for('calendar.index',
                            year=task_date.year, month=task_date.month))


@calendar.route('/tasks/<int:task_id>/status', methods=['POST'])
@login_required
def update_task_status(task_id: int):
    """AJAX: cycle task status → pending → done → skipped → pending."""
    task = CalendarTask.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)

    new_status = request.get_json(silent=True) or {}
    status = new_status.get('status')

    allowed = {'pending', 'done', 'skipped'}
    if status not in allowed:
        # default cycle behaviour
        cycle = {'pending': 'done', 'done': 'skipped', 'skipped': 'pending'}
        status = cycle.get(task.status, 'pending')

    task.status = status
    db.session.commit()
    return jsonify(task.to_dict())


@calendar.route('/tasks/<int:task_id>/delete', methods=['POST'])
@login_required
def delete_task(task_id: int):
    """Delete a task (manual tasks only; roadmap tasks just get skipped)."""
    task = CalendarTask.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    if task.source == 'roadmap':
        task.status = 'skipped'
        db.session.commit()
        if request.is_json:
            return jsonify({'skipped': True})
        flash('Roadmap task marked as skipped.', 'info')
    else:
        db.session.delete(task)
        db.session.commit()
        if request.is_json:
            return jsonify({'deleted': True})
        flash('Task deleted.', 'success')
    return redirect(url_for('calendar.index'))


# ── roadmap views ─────────────────────────────────────────────────────────────

@calendar.route('/roadmaps')
@login_required
def roadmaps():
    """Browse all preset + own AI-generated roadmaps."""
    # preset / global roadmaps
    preset = Roadmap.query.filter_by(source='preset').order_by(Roadmap.title).all()

    # this user's personal AI roadmaps
    personal = Roadmap.query.filter_by(
        source='ai', owner_id=current_user.id
    ).order_by(Roadmap.created_at.desc()).all()

    # IDs the user is already enrolled in
    enrolled_ids = {
        ur.roadmap_id
        for ur in UserRoadmap.query.filter_by(user_id=current_user.id).all()
    }

    return render_template(
        'calendar/roadmaps.html',
        title='Roadmaps',
        preset=preset,
        personal=personal,
        enrolled_ids=enrolled_ids,
    )


@calendar.route('/roadmaps/generate', methods=['POST'])
@login_required
def generate_roadmap():
    """
    Use Groq to generate a personalised roadmap.
    Expects JSON: {topic: str, days: int (7-90)}
    """
    data  = request.get_json(silent=True) or request.form
    topic = (data.get('topic') or '').strip()
    days  = min(max(int(data.get('days', 30)), 7), 90)

    if not topic:
        return jsonify({'error': 'topic is required'}), 400

    try:
        roadmap_data = _groq_generate_roadmap(topic, days)
    except Exception as exc:
        current_app.logger.error("Groq roadmap generation failed: %s", exc)
        return jsonify({'error': 'AI generation failed, please try again.'}), 500

    # persist roadmap + tasks
    roadmap = Roadmap(
        title=roadmap_data['title'],
        topic=topic,
        description=roadmap_data['description'],
        source='ai',
        owner_id=current_user.id,
    )
    db.session.add(roadmap)
    db.session.flush()  # get roadmap.id before adding children

    for t in roadmap_data['tasks']:
        rt = RoadmapTask(
            roadmap_id=roadmap.id,
            day_offset=int(t['day_offset']),
            title=t['title'],
            description=t.get('description', ''),
        )
        db.session.add(rt)

    db.session.commit()

    return jsonify({
        'id':          roadmap.id,
        'title':       roadmap.title,
        'description': roadmap.description,
        'task_count':  len(roadmap_data['tasks']),
    }), 201


@calendar.route('/roadmaps/<int:roadmap_id>/enroll', methods=['POST'])
@login_required
def enroll(roadmap_id: int):
    """Enrol user in a roadmap, starting today, and seed CalendarTask rows."""
    roadmap = Roadmap.query.get_or_404(roadmap_id)

    # access control: only owner can enrol in a personal AI roadmap
    if roadmap.source == 'ai' and roadmap.owner_id != current_user.id:
        abort(403)

    existing = UserRoadmap.query.filter_by(
        user_id=current_user.id, roadmap_id=roadmap_id
    ).first()

    if existing:
        if request.is_json:
            return jsonify({'error': 'already enrolled'}), 409
        flash('You are already enrolled in this roadmap.', 'info')
        return redirect(url_for('calendar.roadmaps'))

    ur = UserRoadmap(
        user_id=current_user.id,
        roadmap_id=roadmap_id,
        start_date=date.today(),
    )
    db.session.add(ur)
    db.session.flush()

    _seed_roadmap_tasks(ur)   # also commits

    if request.is_json:
        return jsonify({
            'enrolled':   True,
            'start_date': ur.start_date.isoformat(),
            'task_count': len(roadmap.tasks),
        }), 200

    flash(f'Enrolled in "{roadmap.title}"! Tasks have been added to your calendar.', 'success')
    return redirect(url_for('calendar.index'))


@calendar.route('/roadmaps/<int:roadmap_id>/unenroll', methods=['POST'])
@login_required
def unenroll(roadmap_id: int):
    """Drop a roadmap. Future (pending) roadmap tasks are deleted; past tasks kept."""
    ur = UserRoadmap.query.filter_by(
        user_id=current_user.id, roadmap_id=roadmap_id
    ).first_or_404()

    today = date.today()
    # delete only future pending tasks to preserve history
    CalendarTask.query.filter(
        CalendarTask.user_id        == current_user.id,
        CalendarTask.source         == 'roadmap',
        CalendarTask.status         == 'pending',
        CalendarTask.date           >  today,
        CalendarTask.roadmap_task_id.in_(
            db.session.query(RoadmapTask.id).filter_by(roadmap_id=roadmap_id)
        ),
        ).delete(synchronize_session='fetch')

    db.session.delete(ur)
    db.session.commit()

    if request.is_json:
        return jsonify({'unenrolled': True})

    flash('You have been unenrolled from the roadmap.', 'info')
    return redirect(url_for('calendar.roadmaps'))
