from __future__ import annotations

import json
import os
import requests
import subprocess
import sys
import tempfile

from flask import (render_template, redirect, url_for,
                   flash, request, jsonify, session, abort)
from flask_login import login_required, current_user

from neurix import db
from neurix.models import ModuleProgress, LevelUnlock
from neurix.learn import learn
from neurix.learn.content import (
    MODULES, LEVEL_META, LEVEL_ORDER,
    get_modules_by_level, get_module_by_id, get_unlock_quiz
)

POINTS_QUIZ_PASS = 5   # points for passing an unlock quiz


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unlocked_levels(user_id: int) -> set:
    """Return set of level names the user has unlocked. Beginner always unlocked."""
    rows = LevelUnlock.query.filter_by(user_id=user_id).all()
    unlocked = {r.level for r in rows}
    unlocked.add("beginner")
    return unlocked


def _completed_module_ids(user_id: int) -> set:
    rows = ModuleProgress.query.filter_by(user_id=user_id, completed=True).all()
    return {r.module_id for r in rows}


def _get_or_create_progress(user_id: int, module_id: str) -> ModuleProgress:
    prog = ModuleProgress.query.filter_by(
        user_id=user_id, module_id=module_id
    ).first()
    if not prog:
        prog = ModuleProgress(user_id=user_id, module_id=module_id)
        db.session.add(prog)
        db.session.commit()
    return prog

def _execute_code(language: str, code: str) -> dict:
    """Execute code locally in a subprocess — no external API needed."""

    if language == "sql":
        code = """
import sqlite3, sys
conn = sqlite3.connect(':memory:')
cur = conn.cursor()
cur.executescript('''
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, years_experience INTEGER);
INSERT INTO employees VALUES (1,'Alice','Engineering',95000,5),(2,'Bob','Marketing',55000,3),(3,'Carol','Engineering',105000,8),(4,'Dave','HR',48000,2),(5,'Eve','Marketing',72000,6),(6,'Frank','Engineering',115000,10),(7,'Grace','HR',52000,4),(8,'Hank','Marketing',61000,5);
CREATE TABLE transactions (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, category TEXT);
INSERT INTO transactions VALUES (1,1,250.0,'Electronics'),(2,2,80.0,'Books'),(3,1,320.0,'Electronics');
CREATE TABLE customers (id INTEGER PRIMARY KEY, age INTEGER, income REAL, churn INTEGER, signup_date TEXT);
INSERT INTO customers VALUES (1,25,50000,0,'2022-06-01'),(2,35,75000,1,'2023-01-15');
''')
user_sql = \"\"\"""" + code.replace('"""', "'''") + """\"\"\"
try:
    cur.execute(user_sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    if cols:
        print(' | '.join(cols))
        print('-' * 40)
    for row in rows:
        print(' | '.join(str(v) for v in row))
except Exception as e:
    print(f'SQL Error: {e}', file=sys.stderr)
"""
        language = "python"

    lang_cmd = {
        "python":     [sys.executable],
        "javascript": ["node"],
    }

    if language not in lang_cmd:
        return {"stdout": "", "stderr": "", "error": f"Unsupported language: {language}"}

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py" if language == "python" else ".js",
            mode="w",
            delete=False,
        ) as f:
            f.write(code)
            fname = f.name

        result = subprocess.run(
            lang_cmd[language] + [fname],
            capture_output=True,
            text=True,
            timeout=5,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error":  result.stderr if result.returncode != 0 else "",
        }

    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "error": "Execution timed out (5s limit)."}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": "", "error": f"Runtime not found: {e}"}
    except Exception as exc:
        return {"stdout": "", "stderr": "", "error": f"Execution error: {exc}"}
    finally:
        try:
            import os; os.unlink(fname)
        except Exception:
            pass


def _check_solution(output: str, checks: list) -> bool:
    """Return True if any check keyword appears in the output."""
    combined = output.lower()
    return any(str(c).lower() in combined for c in checks)


# ── Routes ────────────────────────────────────────────────────────────────────

@learn.route("/learn")
@login_required
def dashboard():
    unlocked = _unlocked_levels(current_user.id)
    completed = _completed_module_ids(current_user.id)

    level_data = []
    for level in LEVEL_ORDER:
        modules = get_modules_by_level(level)
        total = len(modules)
        done = sum(1 for m in modules if m["id"] in completed)
        level_data.append({
            "key": level,
            "meta": LEVEL_META[level],
            "modules": modules,
            "unlocked": level in unlocked,
            "total": total,
            "done": done,
            "pct": int((done / total) * 100) if total else 0,
        })

    return render_template(
        "learn/dashboard.html",
        title="Learn",
        level_data=level_data,
        completed=completed,
    )


@learn.route("/learn/module/<module_id>")
@login_required
def module_page(module_id):
    mod = get_module_by_id(module_id)
    if not mod:
        abort(404)

    unlocked = _unlocked_levels(current_user.id)
    if mod["level"] not in unlocked:
        flash("Complete the unlock quiz to access this level.", "warning")
        return redirect(url_for("learn.dashboard"))

    prog = _get_or_create_progress(current_user.id, module_id)

    return render_template(
        "learn/module.html",
        title=mod["title"],
        mod=mod,
        progress=prog,
    )


@learn.route("/learn/module/<module_id>/complete", methods=["POST"])
@login_required
def complete_module(module_id):
    """Mark module complete via MCQ answer submission."""
    mod = get_module_by_id(module_id)
    if not mod:
        return jsonify({"success": False, "message": "Module not found"}), 404

    chosen = request.json.get("answer", "")
    correct_label = next(
        (o["label"] for o in mod["question"]["options"] if o["correct"]), None
    )

    if chosen.upper() != correct_label:
        return jsonify({"success": False, "message": "Incorrect answer — try again!"})

    prog = _get_or_create_progress(current_user.id, module_id)
    already_done = prog.completed

    if not already_done:
        prog.mark_complete()
        current_user.points += mod["points"]
        db.session.commit()

    return jsonify({
        "success": True,
        "already_done": already_done,
        "points_earned": 0 if already_done else mod["points"],
        "message": "Already completed!" if already_done else f"+{mod['points']} points earned!",
    })


@learn.route("/learn/module/<module_id>/run", methods=["POST"])
@login_required
def run_code(module_id):
    """Execute user code via Piston and check solution."""
    mod = get_module_by_id(module_id)
    if not mod or not mod.get("has_ide"):
        return jsonify({"success": False, "message": "No IDE for this module"}), 400

    data = request.json or {}
    code = data.get("code", "")
    language = data.get("language", mod.get("ide_language", "python"))

    result = _execute_code(language, code)

    output = result["stdout"] + result["stderr"]
    passed = _check_solution(output, mod.get("ide_solution_check", []))

    # Award points if code passes and module not yet completed
    points_earned = 0
    if passed:
        prog = _get_or_create_progress(current_user.id, module_id)
        if not prog.completed:
            prog.mark_complete()
            current_user.points += mod["points"]
            db.session.commit()
            points_earned = mod["points"]

    return jsonify({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "error": result["error"],
        "passed": passed,
        "points_earned": points_earned,
    })


# ── Unlock Quiz ───────────────────────────────────────────────────────────────

@learn.route("/learn/unlock/<level>")
@login_required
def unlock_quiz(level):
    if level not in ["intermediate", "advanced"]:
        abort(404)

    unlocked = _unlocked_levels(current_user.id)
    if level in unlocked:
        flash(f"{LEVEL_META[level]['label']} is already unlocked!", "info")
        return redirect(url_for("learn.dashboard"))

    # Check prerequisite
    prereq = {"intermediate": "beginner", "advanced": "intermediate"}[level]
    if prereq not in unlocked:
        flash(f"Complete {LEVEL_META[prereq]['label']} first.", "warning")
        return redirect(url_for("learn.dashboard"))

    quiz = get_unlock_quiz(level)
    return render_template(
        "learn/unlock_quiz.html",
        title=f"Unlock {LEVEL_META[level]['label']}",
        level=level,
        level_meta=LEVEL_META[level],
        quiz=quiz,
    )


@learn.route("/learn/unlock/<level>/submit", methods=["POST"])
@login_required
def submit_unlock_quiz(level):
    if level not in ["intermediate", "advanced"]:
        abort(404)

    quiz = get_unlock_quiz(level)
    data = request.json or {}

    # Score MCQ
    answers = data.get("mcq_answers", {})   # {question_id: chosen_label}
    mcq_score = 0
    for q in quiz["mcq"]:
        chosen = answers.get(q["id"], "")
        correct = next((o["label"] for o in q["options"] if o["correct"]), None)
        if chosen.upper() == correct:
            mcq_score += 1

    # Check code challenge
    code = data.get("code", "")
    code_lang = quiz["code"]["language"]
    code_result = _execute_code(code_lang, code)
    code_output = code_result["stdout"] + code_result["stderr"]
    code_passed = _check_solution(code_output, quiz["code"]["solution_check"])

    pass_threshold = LEVEL_META[level]["quiz_pass_score"]
    mcq_passed = mcq_score >= pass_threshold

    if mcq_passed and code_passed:
        # Unlock the level
        existing = LevelUnlock.query.filter_by(
            user_id=current_user.id, level=level
        ).first()
        if not existing:
            unlock = LevelUnlock(user_id=current_user.id, level=level)
            db.session.add(unlock)
            current_user.points += POINTS_QUIZ_PASS
            db.session.commit()

        return jsonify({
            "passed": True,
            "mcq_score": mcq_score,
            "code_passed": code_passed,
            "points_earned": POINTS_QUIZ_PASS,
            "message": f"🎉 {LEVEL_META[level]['label']} unlocked! +{POINTS_QUIZ_PASS} points",
        })
    else:
        return jsonify({
            "passed": False,
            "mcq_score": mcq_score,
            "code_passed": code_passed,
            "message": f"MCQ: {mcq_score}/5, Code: {'✓' if code_passed else '✗'}. Need {pass_threshold}/5 MCQ + passing code.",
        })


@learn.route("/learn/unlock/<level>/run_code", methods=["POST"])
@login_required
def quiz_run_code(level):
    """Run code inside the unlock quiz IDE."""
    quiz = get_unlock_quiz(level)
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404

    data = request.json or {}
    code = data.get("code", "")
    language = quiz["code"]["language"]
    result = _execute_code(language, code)

    return jsonify({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "error": result["error"],
    })
