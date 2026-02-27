import json
import os
from datetime import datetime, timezone

from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from groq import Groq

from neurix.quiz import quiz
from neurix import db
from neurix.models import ModuleProgress, QuizAttempt

# ─────────────────────────────────────────────────────────────────────────────
# MODULE CATALOGUE
# Maps every module_id (as stored in ModuleProgress) to its display name,
# topic tags, and which level it belongs to.
# Keep this in sync with your learn/routes.py MODULES definition.
# ─────────────────────────────────────────────────────────────────────────────
MODULE_CATALOGUE = {
    # ── Beginner ──────────────────────────────────────────────────────────────
    "beginner_what_is_ml": {
        "title": "What is Machine Learning?",
        "level": "beginner",
        "tags": ["machine learning", "ml basics", "ai", "introduction"],
    },
    "beginner_types_of_ml": {
        "title": "Types of Machine Learning",
        "level": "beginner",
        "tags": ["supervised learning", "unsupervised learning", "reinforcement learning", "ml basics"],
    },
    "beginner_data_preprocessing": {
        "title": "Data Preprocessing",
        "level": "beginner",
        "tags": ["data preprocessing", "feature engineering", "normalization", "missing values"],
    },
    "beginner_linear_regression": {
        "title": "Linear Regression",
        "level": "beginner",
        "tags": ["linear regression", "regression", "supervised learning"],
    },
    # ── Intermediate ──────────────────────────────────────────────────────────
    "intermediate_decision_trees": {
        "title": "Decision Trees",
        "level": "intermediate",
        "tags": ["decision trees", "classification", "supervised learning"],
    },
    "intermediate_random_forests": {
        "title": "Random Forests",
        "level": "intermediate",
        "tags": ["random forests", "ensemble methods", "bagging"],
    },
    "intermediate_svm": {
        "title": "Support Vector Machines",
        "level": "intermediate",
        "tags": ["svm", "support vector machines", "classification", "kernel"],
    },
    "intermediate_neural_networks": {
        "title": "Neural Networks",
        "level": "intermediate",
        "tags": ["neural networks", "deep learning", "perceptron", "backpropagation"],
    },
    "intermediate_model_evaluation": {
        "title": "Model Evaluation",
        "level": "intermediate",
        "tags": ["model evaluation", "cross validation", "overfitting", "bias variance"],
    },
    # ── Advanced ──────────────────────────────────────────────────────────────
    "advanced_cnn": {
        "title": "Convolutional Neural Networks",
        "level": "advanced",
        "tags": ["cnn", "convolutional neural networks", "computer vision", "deep learning"],
    },
    "advanced_rnn": {
        "title": "Recurrent Neural Networks",
        "level": "advanced",
        "tags": ["rnn", "lstm", "sequence models", "nlp", "time series"],
    },
    "advanced_transformers": {
        "title": "Transformers & Attention",
        "level": "advanced",
        "tags": ["transformers", "attention mechanism", "bert", "nlp", "gpt"],
    },
    "advanced_clustering": {
        "title": "Clustering Algorithms",
        "level": "advanced",
        "tags": ["clustering", "k-means", "unsupervised learning", "dbscan"],
    },
    "advanced_reinforcement": {
        "title": "Reinforcement Learning",
        "level": "advanced",
        "tags": ["reinforcement learning", "q-learning", "policy gradient", "reward"],
    },
}


def _get_completed_module_ids(user_id):
    """Return set of module_ids the user has completed."""
    rows = ModuleProgress.query.filter_by(user_id=user_id, completed=True).all()
    return {r.module_id for r in rows}


def _tags_to_modules(requested_tags):
    """
    Given a list of user-requested tags, return:
      allowed   – modules whose tags overlap AND user completed them
      blocked   – modules whose tags overlap but user hasn't completed them
    Each entry: {module_id, title, level, matched_tags}
    """
    requested_lower = [t.strip().lower() for t in requested_tags if t.strip()]
    completed = _get_completed_module_ids(current_user.id)

    allowed, blocked = [], []
    for mid, meta in MODULE_CATALOGUE.items():
        overlap = [t for t in meta["tags"] if any(r in t or t in r for r in requested_lower)]
        if not overlap:
            continue
        entry = {"module_id": mid, "title": meta["title"],
                 "level": meta["level"], "matched_tags": overlap}
        if mid in completed:
            allowed.append(entry)
        else:
            blocked.append(entry)

    return allowed, blocked


def _generate_quiz(allowed_modules, tags):
    """
    Call Groq to generate 15 questions: 5 easy, 5 medium, 5 hard.
    Returns list of question dicts or raises ValueError.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    module_titles = [m["title"] for m in allowed_modules]
    tag_str = ", ".join(tags)

    system_prompt = """You are an ML quiz generator. Return ONLY valid JSON — no markdown, no explanation.
Schema: {"questions": [{"id":1,"tag":"topic","difficulty":"easy","question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"A","explanation":"..."}]}
Rules:
- Exactly 15 questions total: 5 easy, 5 medium, 5 hard
- difficulty values must be exactly: easy | medium | hard
- answer must be exactly one of: A | B | C | D
- questions must be about the provided topics only
- explanation should be 1-2 sentences
- Each question must have a 'tag' field matching one of the requested topics"""

    user_prompt = f"""Generate a 15-question ML quiz on these topics: {tag_str}
Covered modules: {', '.join(module_titles)}
Distribution: exactly 5 easy, 5 medium, 5 hard questions.
Mix the topics across all difficulties."""

    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )

    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    data = json.loads(raw)
    questions = data.get("questions", [])

    if len(questions) != 15:
        raise ValueError(f"Expected 15 questions, got {len(questions)}")

    # Validate and normalise
    valid_diff = {"easy", "medium", "hard"}
    for q in questions:
        if q.get("difficulty") not in valid_diff:
            raise ValueError(f"Bad difficulty: {q.get('difficulty')}")
        if q.get("answer") not in {"A", "B", "C", "D"}:
            raise ValueError(f"Bad answer: {q.get('answer')}")

    return questions


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
        return jsonify({"error": "Groq API error — check your API key"}), 500

    # Strip answers before sending to client (stored server-side in session or
    # re-validated on submit). We embed them in a hidden signed payload instead.
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