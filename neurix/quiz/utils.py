import json
import os
from groq import Groq
from flask_login import current_user
from datetime import datetime, timezone
from neurix.models import ModuleProgress

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
