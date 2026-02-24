"""
neurix/users/streak_utils.py
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
    activity_type: 'module' | 'quiz' | 'playground' | 'post' | 'login'
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
    Counts ANY activity type. A streak breaks if a day is completely empty.
    """
    rows = ActivityLog.query.filter_by(user_id=user_id).all()
    active: set[date] = {r.date for r in rows}
    if not active:
        return 0, 0

    today = datetime.now(timezone.utc).date()

    # Current streak — start from today, or yesterday if today has no activity yet
    start = today if today in active else today - timedelta(days=1)
    current = 0
    check = start
    while check in active:
        current += 1
        check -= timedelta(days=1)

    # Longest streak
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

def get_heatmap_data(user_id: int) -> Dict:
    """
    Build exactly 53 weeks of data ending on today, starting on the Sunday
    of the week that is 52 weeks before today's Sunday.

    Returns:
        {
          "cells": [...],   # one dict per day in the range
          "total": int,     # total activity count in the year
          "active_days": int,
        }

    Each cell dict:
        date        : "YYYY-MM-DD"
        count       : int   total activities that day (ALL types)
        level       : 0-4
        weekday     : 0=Sun … 6=Sat  (LeetCode uses Sun-start weeks)
        week_index  : int   column index (0 = leftmost)
        is_today    : bool
        is_future   : bool
    """
    today = datetime.now(timezone.utc).date()

    # Align to Sunday (weekday 6 in Python Mon=0 convention → Sunday = 6)
    # LeetCode columns go Sun→Sat. In Python, weekday(): Mon=0, Sun=6
    # Days since last Sunday:
    days_since_sunday = (today.weekday() + 1) % 7   # Sun=0, Mon=1, ..., Sat=6
    this_sunday = today - timedelta(days=days_since_sunday)

    # Start = the Sunday 52 weeks ago
    start_sunday = this_sunday - timedelta(weeks=52)

    # Pull all activity for this user in the range
    rows = ActivityLog.query.filter(
        ActivityLog.user_id == user_id,
        ActivityLog.date >= start_sunday,
        ActivityLog.date <= today,
    ).all()

    # Aggregate ALL types into a single count per date
    date_counts: Dict[date, int] = defaultdict(int)
    for r in rows:
        date_counts[r.date] += r.count

    total_count  = sum(date_counts.values())
    active_days  = len(date_counts)

    # Level thresholds — relative to the user's personal max day
    # so even 1 activity immediately shows a visible green cell
    max_day = max(date_counts.values(), default=1)

    def _level(n: int) -> int:
        if n == 0:
            return 0
        # Ensure at least level 1 for any non-zero count
        ratio = n / max_day
        if ratio <= 0.25: return 1
        if ratio <= 0.50: return 2
        if ratio <= 0.75: return 3
        return 4

    # Build cell list — one entry per calendar day.
    # Columns are strict Sun→Sat weeks. week_index advances only on Saturday.
    # Month boundaries mid-week produce partially-filled columns naturally —
    # the empty slots (no cell data) render as transparent in the SVG,
    # which is exactly the LeetCode gap effect. No extra logic needed.
    cells = []
    current = start_sunday
    week_idx = 0

    while current <= today:
        py_wd = current.weekday()   # Mon=0 … Sun=6
        lc_wd = (py_wd + 1) % 7    # Sun=0, Mon=1, … Sat=6

        count = date_counts.get(current, 0)
        cells.append({
            "date":       current.isoformat(),
            "count":      count,
            "level":      _level(count),
            "weekday":    lc_wd,
            "week_index": week_idx,
            "is_today":   current == today,
        })

        if lc_wd == 6:   # Saturday → start a new column next iteration
            week_idx += 1

        current += timedelta(days=1)

    return {
        "cells":       cells,
        "total":       total_count,
        "active_days": active_days,
    }


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
        yr = today.year
        mo = today.month - i
        while mo <= 0:
            mo += 12
            yr -= 1
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
