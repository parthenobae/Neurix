"""
neurix/users/streak_utils.py
Streak computation and activity logging. Import this anywhere you need to
record or read user activity.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from calendar import month_abbr
from typing import Dict, List, Tuple

from neurix import db
from neurix.models import ActivityLog


# ── Logging ───────────────────────────────────────────────────────────────────

def log_activity(user_id: int, activity_type: str) -> None:
    """
    Record one unit of activity for today (UTC).
    Safe to call many times — increments count if row already exists.

    activity_type values: 'module' | 'quiz' | 'playground' | 'post' | 'login'
    """
    today = datetime.now(timezone.utc).date()
    row = ActivityLog.query.filter_by(
        user_id=user_id,
        date=today,
        activity_type=activity_type,
    ).first()
    if row:
        row.count += 1
    else:
        db.session.add(ActivityLog(
            user_id=user_id,
            date=today,
            activity_type=activity_type,
            count=1,
        ))
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()


# ── Streak ────────────────────────────────────────────────────────────────────

def compute_streak(user_id: int) -> Tuple[int, int]:
    """
    Returns (current_streak, longest_streak) in days.
    A streak is consecutive calendar days with ≥1 activity.
    Today counts even if it is the only active day.
    """
    rows = ActivityLog.query.filter_by(user_id=user_id).all()
    active: set[date] = {r.date for r in rows}
    if not active:
        return 0, 0

    today = datetime.now(timezone.utc).date()

    # ── Current streak ────────────────────────────────
    # Start from today; if today has no activity, try from yesterday
    start = today if today in active else today - timedelta(days=1)
    current = 0
    check = start
    while check in active:
        current += 1
        check -= timedelta(days=1)

    # ── Longest streak ────────────────────────────────
    longest, run, prev = 0, 0, None
    for d in sorted(active):
        if prev is None or d == prev + timedelta(days=1):
            run += 1
        else:
            run = 1
        longest = max(longest, run)
        prev = d

    return current, longest


# ── Heatmap ───────────────────────────────────────────────────────────────────

def get_heatmap_data(user_id: int, weeks: int = 52) -> List[Dict]:
    """
    52-week contribution heatmap cells (Mon–Sun columns).
    Each cell: {date, count, level 0-4, weekday 0=Mon, week_index}

    Levels are based on module completions only, using fixed thresholds so
    a cell turns green the moment the first module is done and never shifts
    retroactively due to other activity types.

    Level thresholds:
        0 → no modules
        1 → 1–2 modules   (light green)
        2 → 3–5 modules
        3 → 6–9 modules
        4 → 10+ modules   (dark green)
    """
    today = datetime.now(timezone.utc).date()
    # Roll back to the Monday of (today - 52 weeks)
    start = today - timedelta(weeks=weeks)
    start -= timedelta(days=start.weekday())   # snap to Monday

    rows = ActivityLog.query.filter(
        ActivityLog.user_id == user_id,
        ActivityLog.activity_type == 'module',   # ← modules only
        ActivityLog.date >= start,
    ).all()

    date_counts: Dict[date, int] = defaultdict(int)
    for r in rows:
        date_counts[r.date] += r.count

    # Fixed absolute thresholds — levels never shift due to other days/types
    def _level(n: int) -> int:
        if n == 0:  return 0
        if n <= 2:  return 1
        if n <= 5:  return 2
        if n <= 9:  return 3
        return 4

    cells, current, week_idx = [], start, 0
    while current <= today:
        count = date_counts.get(current, 0)
        level = _level(count)

        cells.append({
            "date":       current.isoformat(),
            "count":      count,
            "level":      level,
            "weekday":    current.weekday(),
            "week_index": week_idx,
        })

        if current.weekday() == 6:   # Sunday → advance week counter
            week_idx += 1
        current += timedelta(days=1)

    return cells


# ── Summary helpers ───────────────────────────────────────────────────────────

def get_activity_summary(user_id: int) -> Dict[str, int]:
    """Total per activity_type over the last 365 days."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=365)
    rows = ActivityLog.query.filter(
        ActivityLog.user_id == user_id,
        ActivityLog.date >= cutoff,
    ).all()
    result: Dict[str, int] = defaultdict(int)
    for r in rows:
        result[r.activity_type] += r.count
    return dict(result)


def get_monthly_counts(user_id: int, months: int = 6) -> List[Dict]:
    """Total activity per month for the last N months."""
    today = datetime.now(timezone.utc).date()
    result = []
    for i in range(months - 1, -1, -1):
        yr  = today.year
        mo  = today.month - i
        while mo <= 0:
            mo += 12; yr -= 1
        first = date(yr, mo, 1)
        last  = date(yr, mo + 1, 1) - timedelta(days=1) if mo < 12 else date(yr, 12, 31)
        rows  = ActivityLog.query.filter(
            ActivityLog.user_id == user_id,
            ActivityLog.date >= first,
            ActivityLog.date <= last,
        ).all()
        result.append({
            "month": month_abbr[mo],
            "total": sum(r.count for r in rows),
        })
    return result
