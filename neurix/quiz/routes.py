import json

from flask import render_template, request, jsonify
from flask_login import login_required, current_user

from neurix.models import QuizAttempt
from neurix.quiz import quiz
from neurix import db
from neurix.quiz.utils import (
    _get_completed_module_ids,
    _tags_to_modules,
    _generate_quiz,
    MODULE_CATALOGUE
)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────


@quiz.route('/quiz')
@login_required
def index():
    """Landing page — tag selector."""
    completed = _get_completed_module_ids(current_user.id)
    # Build tag cloud from completed modules only
    available_tags = set()
    for mid in completed:
        if mid in MODULE_CATALOGUE:
            available_tags.update(MODULE_CATALOGUE[mid]["tags"])

    # Past attempts for efficiency chart
    attempts = (
        QuizAttempt.query
        .filter_by(user_id=current_user.id)
        .order_by(QuizAttempt.attempted_at.asc())
        .limit(10)
        .all()
    )
    history = [
        {
            "date": a.attempted_at.strftime("%b %d"),
            "score": a.score,
            "total": a.total,
            "pct": round(a.score / a.total * 100),
            "tags": a.tags,
            "time": a.time_taken,
        }
        for a in attempts
    ]

    return render_template(
        'quiz/index.html',
        title='Quiz',
        available_tags=sorted(available_tags),
        history=history,
    )


@quiz.route('/quiz/validate_tags', methods=['POST'])
@login_required
def validate_tags():
    """
    AJAX: given tags, return which modules are allowed vs blocked.
    """
    data = request.get_json(force=True)
    tags = data.get("tags", [])
    if not tags:
        return jsonify({"error": "No tags provided"}), 400

    allowed, blocked = _tags_to_modules(tags)
    return jsonify({
        "allowed":  allowed,
        "blocked":  blocked,
        "can_start": len(allowed) > 0,
    })


@quiz.route('/quiz/generate', methods=['POST'])
@login_required
def generate():
    """
    AJAX: generate quiz questions via Groq.
    Returns 15 questions (without the answer key — sent separately on submit).
    """
    data = request.get_json(force=True)
    tags = data.get("tags", [])
    if not tags:
        return jsonify({"error": "No tags provided"}), 400

    allowed, blocked = _tags_to_modules(tags)
    if not allowed:
        return jsonify({"error": "No completed modules match these tags"}), 403

    try:
        questions = _generate_quiz(allowed, tags)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Groq error: {str(e)}"}), 500

    # Strip answers before sending to client (stored server-side in session or
    # re-validated on submit). We embed them in a hidden signed payload instead
    client_questions = []
    answer_key = {}
    for q in questions:
        answer_key[str(q["id"])] = {
            "answer":      q["answer"],
            "explanation": q.get("explanation", ""),
            "tag":         q.get("tag", ""),
            "difficulty":  q["difficulty"],
        }
        client_questions.append({
            "id":         q["id"],
            "tag":        q.get("tag", ""),
            "difficulty": q["difficulty"],
            "question":   q["question"],
            "options":    q["options"],
        })

    return jsonify({
        "questions":  client_questions,
        "answer_key": answer_key,   # sent to client, validated on submit
        "blocked":    blocked,
    })


@quiz.route('/quiz/submit', methods=['POST'])
@login_required
def submit():
    """
    AJAX: score the quiz, persist attempt, return full results.
    """
    data = request.get_json(force=True)
    answers    = data.get("answers", {})      # {question_id: chosen_option}
    answer_key = data.get("answer_key", {})   # from generate response
    tags       = data.get("tags", [])
    time_taken = data.get("time_taken", None)

    if not answer_key:
        return jsonify({"error": "Missing answer key"}), 400

    results = []
    score = 0
    diff_correct = {"easy": 0, "medium": 0, "hard": 0}
    diff_total   = {"easy": 0, "medium": 0, "hard": 0}
    tag_correct  = {}
    tag_total    = {}

    for qid, key in answer_key.items():
        correct  = key["answer"]
        chosen   = answers.get(qid, "")
        is_right = chosen == correct
        diff     = key["difficulty"]
        tag      = key.get("tag", "General")

        diff_total[diff] = diff_total.get(diff, 0) + 1
        tag_total[tag]   = tag_total.get(tag, 0) + 1

        if is_right:
            score += 1
            diff_correct[diff] = diff_correct.get(diff, 0) + 1
            tag_correct[tag]   = tag_correct.get(tag, 0) + 1

        results.append({
            "id":          qid,
            "chosen":      chosen,
            "correct":     correct,
            "is_correct":  is_right,
            "explanation": key.get("explanation", ""),
            "tag":         tag,
            "difficulty":  diff,
        })

    # Weak topics: tags where correct < 50%
    weak_tags = [
        tag for tag in tag_total
        if (tag_correct.get(tag, 0) / tag_total[tag]) < 0.5
    ]

    # Persist attempt
    attempt = QuizAttempt(
        user_id=current_user.id,
        tags=", ".join(tags),
        score=score,
        total=15,
        time_taken=time_taken,
        difficulty_breakdown=json.dumps(diff_correct),
        tag_breakdown=json.dumps(tag_correct),
    )
    db.session.add(attempt)

    # Award points: 2 pts per correct answer
    current_user.points += score * 2
    db.session.commit()

    # Build difficulty breakdown for response
    diff_breakdown = {
        d: {"correct": diff_correct.get(d, 0), "total": diff_total.get(d, 0)}
        for d in ["easy", "medium", "hard"]
    }

    return jsonify({
        "score":             score,
        "total":             15,
        "pct":               round(score / 15 * 100),
        "points_earned":     score * 2,
        "results":           results,
        "diff_breakdown":    diff_breakdown,
        "tag_breakdown":     {t: {"correct": tag_correct.get(t, 0), "total": tag_total[t]} for t in tag_total},
        "weak_tags":         weak_tags,
        "time_taken":        time_taken,
    })
