from __future__ import annotations

import json
import requests

from flask import (render_template, redirect, url_for,
                   flash, request, jsonify, session, abort)
from flask_login import login_required, current_user
from neurix.users.streak_utils import log_activity

from neurix import db
from neurix.models import ModuleProgress, LevelUnlock, ChatMessage
from neurix.learn import learn
from neurix.learn.content import (
    MODULES, LEVEL_META, LEVEL_ORDER,
    get_modules_by_level, get_module_by_id, get_unlock_quiz
)
from neurix.learn.utils import (
    _unlocked_levels, _completed_module_ids,
    _get_or_create_progress, _execute_code,
    _check_solution
)

POINTS_QUIZ_PASS = 5   # points for passing an unlock quiz


# ── Routes ──────────────────────────────────────────────────────────────────
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

    # Return early if already done — skip IDE gate entirely
    if prog.completed:
        return jsonify({
            "success": True,
            "already_done": True,
            "points_earned": 0,
            "message": "Already completed!",
        })

    # For modules with a coding challenge, code must be passed before MCQ counts
    if mod.get("has_ide") and not session.get(f"code_passed_{module_id}"):
        return jsonify({
            "success": False,
            "message": "Complete the coding challenge first, then answer the quiz."
        })

    prog.mark_complete()
    current_user.points += mod["points"]
    db.session.commit()
    log_activity(current_user.id, 'module')
    session.pop(f"code_passed_{module_id}", None)

    return jsonify({
        "success": True,
        "already_done": False,
        "points_earned": mod["points"],
        "message": f"+{mod['points']} points earned!",
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

    # Track code-passed in session so complete_module can check it.
    # Do NOT mark the module complete here — completion requires BOTH
    # the code challenge AND the MCQ to be submitted correctly.
    if passed:
        session[f"code_passed_{module_id}"] = True

    return jsonify({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "error": result["error"],
        "passed": passed,
        "points_earned": 0,   # points awarded only after MCQ in complete_module
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
            log_activity(current_user.id, 'quiz')

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


# ── Chatbot ───────────────────────────────────────────────────────────────────

def _build_system_prompt(mod: dict) -> str:
    """
    Build a tight system prompt that:
    - Knows the module's full theory and quiz
    - Refuses to give direct answers
    - Uses Socratic hinting only
    """
    import re
    # Strip HTML tags from theory for cleaner context
    theory_plain = re.sub(r'<[^>]+>', ' ', mod.get('theory', ''))
    theory_plain = re.sub(r'\s+', ' ', theory_plain).strip()

    # Build quiz context string
    q = mod.get('question', {})
    quiz_context = ""
    if q:
        opts = "  ".join(
            f"{o['label']}) {o['text']}" for o in q.get('options', [])
        )
        quiz_context = f"\nMCQ Question the student must answer: {q.get('text', '')}\nOptions: {opts}"

    # Build code challenge context if present
    code_context = ""
    if mod.get('has_ide'):
        code_context = (
            f"\nCoding challenge the student must solve: {mod.get('challenge_description', '')}"
            f"\nStarter code:\n{mod.get('ide_starter', '')}"
        )

    return f"""You are a focused ML tutor for the module: "{mod['title']}".

MODULE CONTENT:
{theory_plain}
{quiz_context}
{code_context}

YOUR STRICT RULES:
1. NEVER give the direct answer to the MCQ question or the coding challenge — not even partially.
2. Instead, ask the student a guiding question that nudges them toward the answer.
3. If they are stuck on code, give a conceptual hint about WHAT to think about, not HOW to write it.
4. Keep responses short — 3 to 5 sentences maximum.
5. If the student asks something outside this module's topic, gently redirect them back.
6. Be encouraging and patient. Use plain English, avoid jargon unless explaining it.
7. You may explain theory concepts from the module content freely — just never reveal quiz/challenge answers.
"""


@learn.route("/learn/module/<module_id>/chat/history", methods=["GET"])
@login_required
def chat_history(module_id):
    """Return the last 40 messages for this user + module."""
    mod = get_module_by_id(module_id)
    if not mod:
        return jsonify({"error": "Module not found"}), 404

    messages = (
        ChatMessage.query
        .filter_by(user_id=current_user.id, module_id=module_id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(40)
        .all()
    )
    return jsonify({"messages": [m.to_dict() for m in messages]})


@learn.route("/learn/module/<module_id>/chat/message", methods=["POST"])
@login_required
def chat_message(module_id):
    """
    Accept a user message, call Groq API, stream reply back,
    and persist both turns to ChatMessage table.
    """
    from flask import current_app, Response, stream_with_context
    from groq import Groq

    mod = get_module_by_id(module_id)
    if not mod:
        return jsonify({"error": "Module not found"}), 404

    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    # ── Save user message ────────────────────────────────
    user_msg = ChatMessage(
        user_id=current_user.id,
        module_id=module_id,
        role="user",
        content=user_text,
    )
    db.session.add(user_msg)
    db.session.commit()

    # ── Build conversation history for Groq ─────────────
    history = (
        ChatMessage.query
        .filter_by(user_id=current_user.id, module_id=module_id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(20)          # last 20 turns as context window
        .all()
    )
    groq_messages = [
        {"role": m.role, "content": m.content} for m in history
    ]

    system_prompt = _build_system_prompt(mod)

    # ── Call Groq API with streaming ─────────────────────
    api_key = current_app.config.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    client = Groq(api_key=api_key)

    def generate():
        full_reply = []
        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *groq_messages,
                ],
                max_tokens=512,
                temperature=0.5,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply.append(delta)
                    # Server-Sent Events format
                    yield f"data: {json.dumps({'token': delta})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # ── Persist assistant reply after streaming ends ─
        if full_reply:
            assistant_msg = ChatMessage(
                user_id=current_user.id,
                module_id=module_id,
                role="assistant",
                content="".join(full_reply),
            )
            db.session.add(assistant_msg)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # disable Nginx buffering for SSE
        },
    )


@learn.route("/learn/module/<module_id>/chat/clear", methods=["POST"])
@login_required
def chat_clear(module_id):
    """Delete all chat history for this user + module."""
    mod = get_module_by_id(module_id)
    if not mod:
        return jsonify({"error": "Module not found"}), 404

    ChatMessage.query.filter_by(
        user_id=current_user.id, module_id=module_id
    ).delete()
    db.session.commit()
    return jsonify({"success": True})
