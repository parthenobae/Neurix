from __future__ import annotations

import json
import logging
import os
import random
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from flask import request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room

from neurix import db, socketio
from neurix.models import User

try:
    from groq import Groq
    _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception:
    _groq_client = None

log = logging.getLogger(__name__)

# ── Fallback questions ─────────────────────────────────────────────────────────
_FALLBACK_QUESTIONS = [
    {
        "question": "Which algorithm is commonly used for binary classification and outputs probabilities using a sigmoid function?",
        "answer": "Logistic Regression",
        "distractors": ["Linear Regression", "Decision Tree", "K-Nearest Neighbors"],
    },
    {
        "question": "In gradient descent, what parameter controls the step size of each update?",
        "answer": "Learning Rate",
        "distractors": ["Momentum", "Batch Size", "Weight Decay"],
    },
    {
        "question": "Which technique randomly drops neurons during training to reduce overfitting?",
        "answer": "Dropout",
        "distractors": ["Batch Normalisation", "L2 Regularisation", "Early Stopping"],
    },
    {
        "question": "What is the name of the process of adjusting model weights using the chain rule of calculus?",
        "answer": "Backpropagation",
        "distractors": ["Forward Pass", "Gradient Clipping", "Weight Initialisation"],
    },
    {
        "question": "Which unsupervised learning algorithm groups data points into k clusters?",
        "answer": "K-Means Clustering",
        "distractors": ["DBSCAN", "PCA", "Linear Discriminant Analysis"],
    },
    {
        "question": "Which algorithm builds an ensemble of decision trees using random feature subsets?",
        "answer": "Random Forest",
        "distractors": ["AdaBoost", "Support Vector Machine", "Naive Bayes"],
    },
    {
        "question": "Which activation function outputs values strictly between 0 and 1?",
        "answer": "Sigmoid",
        "distractors": ["ReLU", "Tanh", "Softmax"],
    },
    {
        "question": "What is the name of the optimisation algorithm that adapts learning rates per parameter?",
        "answer": "Adam",
        "distractors": ["SGD", "RMSProp", "Adagrad"],
    },
    {
        "question": "Which dimensionality reduction technique projects data onto directions of maximum variance?",
        "answer": "PCA",
        "distractors": ["t-SNE", "UMAP", "LDA"],
    },
    {
        "question": "Which metric is often preferred over accuracy for imbalanced binary datasets?",
        "answer": "F1 Score",
        "distractors": ["Mean Squared Error", "R² Score", "Log Loss"],
    },
]

TOTAL_ROUNDS     = 10
POINTS_PER_ROUND = 1
DISCONNECT_BONUS = 3
LABELS = ["A", "B", "C", "D"]

# ── AI question generation ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a machine learning quiz question generator. "
    "You respond ONLY with valid raw JSON arrays — no markdown, no explanation, no backticks."
)

_USER_PROMPT = """\
Generate {n} unique multiple-choice questions for a competitive 1v1 machine learning quiz game.

Rules:
- Test ML/AI/data-science knowledge at undergraduate level.
- Each question must have exactly 1 correct answer and 3 plausible but wrong distractors.
- All {n} questions must cover different topics.
- Keep question text concise (1-2 sentences max).
- Keep each option text to 10 words or fewer.

Return a JSON array where each element has exactly these keys:
  "question"    : string
  "answer"      : string  (correct option)
  "distractors" : array of exactly 3 strings (wrong options)

Example:
[{{"question":"What does learning rate control in gradient descent?","answer":"Step size of each weight update","distractors":["Number of training epochs","Size of the training batch","Depth of the neural network"]}}]
"""


def _generate_questions_via_ai(n: int = TOTAL_ROUNDS) -> List[Dict]:
    if _groq_client is None:
        log.warning("Groq client not available — using fallback questions.")
        return []
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_PROMPT.format(n=n)},
            ],
            temperature=0.8,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        questions = json.loads(raw)

        validated = []
        for q in questions:
            if (
                isinstance(q, dict)
                and isinstance(q.get("question"), str)
                and isinstance(q.get("answer"), str)
                and isinstance(q.get("distractors"), list)
                and len(q["distractors"]) == 3
                and all(isinstance(d, str) for d in q["distractors"])
            ):
                validated.append({
                    "question":    q["question"].strip(),
                    "answer":      q["answer"].strip(),
                    "distractors": [d.strip() for d in q["distractors"]],
                })

        if len(validated) >= n:
            return validated[:n]

        log.warning("Groq returned %d valid questions, expected %d. Using fallback.", len(validated), n)
        return []

    except Exception as exc:
        log.exception("AI question generation failed: %s", exc)
        return []


def _pick_questions() -> List[Dict]:
    questions = _generate_questions_via_ai(TOTAL_ROUNDS)
    if not questions:
        pool = _FALLBACK_QUESTIONS.copy()
        random.shuffle(pool)
        questions = pool[:TOTAL_ROUNDS]
    return questions


# ── Option builder ─────────────────────────────────────────────────────────────

def _build_options(question: Dict) -> List[Dict]:
    choices = [{"text": question["answer"], "correct": True}]
    for d in question["distractors"]:
        choices.append({"text": d, "correct": False})
    random.shuffle(choices)
    return [
        {"label": LABELS[i], "text": c["text"], "correct": c["correct"]}
        for i, c in enumerate(choices)
    ]


# ── Matchmaker ─────────────────────────────────────────────────────────────────

class Matchmaker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.waiting_player: Optional[Dict] = None
        self.rooms: Dict[str, Dict] = {}

    def _award_points(self, user_id: int, amount: int) -> None:
        user = User.query.get(user_id)
        if not user:
            return
        try:
            user.points += amount
            db.session.commit()
        except Exception:
            db.session.rollback()

    def _scores_payload(self, room: Dict) -> Dict:
        return {p["username"]: room["scores"][p["sid"]] for p in room["players"]}

    def _question_payload(self, room: Dict) -> Dict:
        opts = room["current_options"]
        return {
            "question": room["current_question"]["question"],
            "options":  [{"label": o["label"], "text": o["text"]} for o in opts],
        }

    def _is_correct(self, room: Dict, label: str) -> bool:
        label = label.strip().upper()
        return any(o["label"] == label and o["correct"] for o in room["current_options"])

    def _correct_label(self, room: Dict) -> str:
        for o in room["current_options"]:
            if o["correct"]:
                return o["label"]
        return ""

    def _advance_round(self, room: Dict) -> None:
        room["round"] += 1
        room["round_winner_sid"] = None

        if room["round"] > TOTAL_ROUNDS:
            self._finish_match(room, reason="all_rounds_complete")
            return

        room["current_question"] = room["questions"][room["round"] - 1]
        room["current_options"]  = _build_options(room["current_question"])

        emit(
            "next_round",
            {
                "round":        room["round"],
                "total_rounds": TOTAL_ROUNDS,
                "scores":       self._scores_payload(room),
                **self._question_payload(room),
            },
            to=room["id"],
        )

    def _finish_match(self, room: Dict, reason: str) -> None:
        room["active"] = False
        scores  = room["scores"]
        players = room["players"]

        score_a = scores[players[0]["sid"]]
        score_b = scores[players[1]["sid"]]

        if   score_a > score_b: overall_winner = players[0]
        elif score_b > score_a: overall_winner = players[1]
        else:                   overall_winner = None

        for player in players:
            pts = scores[player["sid"]]
            if pts > 0:
                self._award_points(player["user_id"], pts)

        emit(
            "match_ended",
            {
                "reason": reason,
                "scores": self._scores_payload(room),
                "winner": overall_winner["username"] if overall_winner else None,
                "draw":   overall_winner is None,
            },
            to=room["id"],
        )
        self.rooms.pop(room["id"], None)

    def join_queue(self, sid: str, user_id: int, username: str) -> None:
        with self._lock:
            if self.waiting_player and self.waiting_player["sid"] == sid:
                emit("queue_status", {"message": "You are already waiting for an opponent."})
                return

            if self._find_room_by_sid(sid):
                emit("queue_status", {"message": "You are already in a match."})
                return

            if self.waiting_player is None:
                self.waiting_player = {"sid": sid, "user_id": user_id, "username": username}
                emit("queue_status", {"message": "Waiting for an opponent..."})
                return

            if self.waiting_player["user_id"] == user_id:
                emit("queue_status", {"message": "You cannot play against yourself."})
                return

            player_one = self.waiting_player
            player_two = {"sid": sid, "user_id": user_id, "username": username}
            self.waiting_player = None

            emit("queue_status", {"message": "Opponent found! Generating questions with AI..."}, to=sid)
            emit("queue_status", {"message": "Opponent found! Generating questions with AI..."}, to=player_one["sid"])

            room_id    = f"playground-{uuid4().hex}"
            questions  = _pick_questions()
            first_q    = questions[0]
            first_opts = _build_options(first_q)

            room_state = {
                "id":               room_id,
                "players":          [player_one, player_two],
                "questions":        questions,
                "current_question": first_q,
                "current_options":  first_opts,
                "round":            1,
                "round_winner_sid": None,
                "scores":           {player_one["sid"]: 0, player_two["sid"]: 0},
                "active":           True,
            }
            self.rooms[room_id] = room_state

            join_room(room_id, sid=player_one["sid"])
            join_room(room_id, sid=player_two["sid"])

            emit(
                "match_found",
                {
                    "room_id":      room_id,
                    "round":        1,
                    "total_rounds": TOTAL_ROUNDS,
                    "players":      [player_one["username"], player_two["username"]],
                    "scores":       {player_one["username"]: 0, player_two["username"]: 0},
                    "question":     first_q["question"],
                    "options":      [{"label": o["label"], "text": o["text"]} for o in first_opts],
                    "ai_generated": _groq_client is not None,
                },
                to=room_id,
            )

    def submit_answer(self, sid: str, label: str) -> None:
        with self._lock:
            room = self._find_room_by_sid(sid)
            if not room:
                emit("answer_result", {"correct": False, "message": "No active match found."})
                return
            if not room["active"]:
                emit("answer_result", {"correct": False, "message": "Match already finished."})
                return
            if not label:
                emit("answer_result", {"correct": False, "message": "No option selected."})
                return
            if room["round_winner_sid"] is not None:
                emit("answer_result", {"correct": False, "message": "Round already won."})
                return

            if self._is_correct(room, label):
                room["round_winner_sid"] = sid
                room["scores"][sid] += POINTS_PER_ROUND
                winner = next(p for p in room["players"] if p["sid"] == sid)
                emit(
                    "round_ended",
                    {
                        "round":          room["round"],
                        "round_winner":   winner["username"],
                        "correct_label":  self._correct_label(room),
                        "correct_answer": room["current_question"]["answer"],
                        "scores":         self._scores_payload(room),
                    },
                    to=room["id"],
                )
                self._advance_round(room)
            else:
                emit(
                    "answer_result",
                    {"correct": False, "message": "Wrong answer!", "chosen": label.strip().upper()},
                    to=sid,
                )

    def disconnect(self, sid: str) -> None:
        with self._lock:
            if self.waiting_player and self.waiting_player["sid"] == sid:
                self.waiting_player = None
                return

            room = self._find_room_by_sid(sid)
            if not room or not room["active"]:
                return

            room["active"] = False
            leaver    = next(p for p in room["players"] if p["sid"] == sid)
            remaining = next(p for p in room["players"] if p["sid"] != sid)

            self._award_points(remaining["user_id"], room["scores"][remaining["sid"]] + DISCONNECT_BONUS)
            earned = room["scores"][leaver["sid"]]
            if earned > 0:
                self._award_points(leaver["user_id"], earned)

            leave_room(room["id"], sid=leaver["sid"])
            emit(
                "match_ended",
                {
                    "reason":           "opponent_disconnected",
                    "scores":           self._scores_payload(room),
                    "winner":           remaining["username"],
                    "draw":             False,
                    "disconnect_bonus": DISCONNECT_BONUS,
                },
                to=remaining["sid"],
            )
            self.rooms.pop(room["id"], None)

    def _find_room_by_sid(self, sid: str) -> Optional[Dict]:
        for room in self.rooms.values():
            for player in room["players"]:
                if player["sid"] == sid:
                    return room
        return None


matchmaker = Matchmaker()


# ── Socket event handlers ──────────────────────────────────────────────────────

@socketio.on("join_playground")
def handle_join_playground():
    if not current_user.is_authenticated:
        emit("queue_status", {"message": "Please log in to use the playground."})
        return
    matchmaker.join_queue(request.sid, current_user.id, current_user.username)


@socketio.on("submit_answer")
def handle_submit_answer(data):
    label = ""
    if isinstance(data, dict):
        label = data.get("label", "")
    matchmaker.submit_answer(request.sid, label)


@socketio.on("disconnect")
def handle_disconnect():
    matchmaker.disconnect(request.sid)
