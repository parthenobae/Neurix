from __future__ import annotations

import random
import time
import threading
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from flask import request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room

from neurix import db, socketio
from neurix.models import User, ModuleProgress
from neurix.users.streak_utils import log_activity


QUESTIONS = [
    {
        "question": "Which algorithm is commonly used for binary classification and outputs probabilities using a sigmoid function?",
        "answer": "Logistic Regression",
        "distractors": ["Linear Regression", "Decision Tree", "K-Nearest Neighbors"],
    },
    {
        "question": "What does overfitting mean in machine learning?",
        "answer": "The model memorises training data and fails to generalise",
        "distractors": [
            "The model is too simple to capture patterns",
            "The model converges too slowly",
            "The model uses too few features",
        ],
    },
    {
        "question": "Which validation strategy repeatedly splits data into training and validation folds?",
        "answer": "K-Fold Cross Validation",
        "distractors": ["Hold-Out Validation", "Leave-One-Out", "Bootstrap Sampling"],
    },
    {
        "question": "In gradient descent, what parameter controls the step size of each update?",
        "answer": "Learning Rate",
        "distractors": ["Momentum", "Batch Size", "Weight Decay"],
    },
    {
        "question": "Which metric is often preferred over accuracy for imbalanced binary datasets?",
        "answer": "F1 Score",
        "distractors": ["Mean Squared Error", "R² Score", "Log Loss"],
    },
    {
        "question": "What type of neural network layer connects every neuron to every neuron in the next layer?",
        "answer": "Fully Connected (Dense) Layer",
        "distractors": ["Convolutional Layer", "Pooling Layer", "Recurrent Layer"],
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
        "distractors": ["DBSCAN", "Principal Component Analysis", "Linear Discriminant Analysis"],
    },
    {
        "question": "Which algorithm builds an ensemble of decision trees using random feature subsets?",
        "answer": "Random Forest",
        "distractors": ["AdaBoost", "Support Vector Machine", "Naive Bayes"],
    },
    {
        "question": "What term describes the error caused by overly simple assumptions in a learning algorithm?",
        "answer": "Bias",
        "distractors": ["Variance", "Entropy", "Regularisation"],
    },
    {
        "question": "Which activation function outputs values strictly between 0 and 1, used in binary classification output layers?",
        "answer": "Sigmoid",
        "distractors": ["ReLU", "Tanh", "Softmax"],
    },
    {
        "question": "What is the name of the optimisation algorithm that adapts learning rates for each parameter individually?",
        "answer": "Adam",
        "distractors": ["SGD", "RMSProp", "Adagrad"],
    },
    {
        "question": "Which dimensionality reduction technique projects data onto directions of maximum variance?",
        "answer": "Principal Component Analysis (PCA)",
        "distractors": ["t-SNE", "UMAP", "Linear Discriminant Analysis"],
    },
    {
        "question": "What is the purpose of a confusion matrix in classification tasks?",
        "answer": "To show the counts of true/false positives and negatives",
        "distractors": [
            "To measure the distance between class centroids",
            "To visualise feature correlations",
            "To plot the learning curve",
        ],
    },
    {
        "question": "Which loss function is standard for multi-class classification with softmax output?",
        "answer": "Categorical Cross-Entropy",
        "distractors": ["Mean Squared Error", "Hinge Loss", "Huber Loss"],
    },
    {
        "question": "What does the ROC curve plot?",
        "answer": "True Positive Rate vs False Positive Rate",
        "distractors": [
            "Precision vs Recall",
            "Loss vs Epochs",
            "Accuracy vs Model Complexity",
        ],
    },
    {
        "question": "Which technique scales each feature to have zero mean and unit variance?",
        "answer": "Standardisation (Z-score normalisation)",
        "distractors": ["Min-Max Scaling", "Log Transformation", "One-Hot Encoding"],
    },
]

TOTAL_ROUNDS     = 10
POINTS_PER_ROUND = 1
DISCONNECT_BONUS = 3
ROUND_TIMEOUT    = 30      # seconds before a round is declared no-winner
LABELS           = ["A", "B", "C", "D"]


def _completed_count(user_id: int) -> int:
    return ModuleProgress.query.filter_by(user_id=user_id, completed=True).count()


def _build_options(question: Dict) -> List[Dict]:
    choices = [{"text": question["answer"], "correct": True}]
    for d in question["distractors"]:
        choices.append({"text": d, "correct": False})
    random.shuffle(choices)
    return [{"label": LABELS[i], "text": c["text"], "correct": c["correct"]}
            for i, c in enumerate(choices)]


class Matchmaker:
    def __init__(self) -> None:
        self._lock  = Lock()
        self.waiting_queue: List[Dict] = []
        self.rooms:         Dict[str, Dict] = {}

    # ── Points & activity ──────────────────────────────────────────────────

    def _award_points(self, user_id: int, amount: int) -> None:
        user = User.query.get(user_id)
        if not user:
            return
        try:
            user.points += amount
            db.session.commit()
        except Exception:
            db.session.rollback()

    # ── Payload helpers ────────────────────────────────────────────────────

    def _scores_payload(self, room: Dict) -> Dict:
        return {p["username"]: room["scores"][p["sid"]] for p in room["players"]}

    def _question_payload(self, room: Dict) -> Dict:
        opts = room["current_options"]
        return {
            "question": room["current_question"]["question"],
            "options":  [{"label": o["label"], "text": o["text"]} for o in opts],
        }

    def _is_correct(self, room: Dict, label: str) -> bool:
        return any(
            o["label"] == label.strip().upper() and o["correct"]
            for o in room["current_options"]
        )

    def _correct_label(self, room: Dict) -> str:
        for o in room["current_options"]:
            if o["correct"]:
                return o["label"]
        return ""

    # ── Round timeout ──────────────────────────────────────────────────────

    def _start_round_timer(self, room: Dict) -> None:
        """
        Fire a background thread that expires the round after ROUND_TIMEOUT
        seconds if neither player has answered correctly yet.
        """
        round_at_start = room["round"]
        room_id        = room["id"]

        def _expire():
            time.sleep(ROUND_TIMEOUT)
            with self._lock:
                room = self.rooms.get(room_id)
                # Only act if the match is still alive and still on the same round
                if not room or not room["active"]:
                    return
                if room["round"] != round_at_start:
                    return
                # Neither player answered correctly — advance with no winner
                self._resolve_round(room, winner_sid=None, timed_out=True)

        t = threading.Thread(target=_expire, daemon=True)
        t.start()

    # ── Core round logic ───────────────────────────────────────────────────

    def _resolve_round(self, room: Dict, winner_sid: Optional[str], timed_out: bool = False) -> None:
        """
        Called when a round ends — either someone answered correctly,
        both players used their one attempt and were wrong, or time ran out.
        Emits round_ended then advances.
        """
        correct_lbl = self._correct_label(room)

        if winner_sid:
            winner = next(p for p in room["players"] if p["sid"] == winner_sid)
            room["scores"][winner_sid] += POINTS_PER_ROUND
            winner_name = winner["username"]
            reason_msg  = f"{winner_name} answered correctly!"
        else:
            winner_name = None
            reason_msg  = "Time's up — no one answered correctly." if timed_out else "Both players answered incorrectly."

        emit(
            "round_ended",
            {
                "round":          room["round"],
                "round_winner":   winner_name,
                "correct_label":  correct_lbl,
                "correct_answer": room["current_question"]["answer"],
                "scores":         self._scores_payload(room),
                "timed_out":      timed_out,
                "message":        reason_msg,
            },
            to=room["id"],
        )
        self._advance_round(room)

    def _advance_round(self, room: Dict) -> None:
        room["round"]         += 1
        room["answered_sids"]  = set()    # reset per-round attempt tracking
        room["question_sent_at"] = None

        if room["round"] > TOTAL_ROUNDS:
            self._finish_match(room, reason="all_rounds_complete")
            return

        room["current_question"] = room["questions"][room["round"] - 1]
        room["current_options"]  = _build_options(room["current_question"])
        room["question_sent_at"] = time.time()

        emit(
            "next_round",
            {
                "round":        room["round"],
                "total_rounds": TOTAL_ROUNDS,
                "scores":       self._scores_payload(room),
                "timeout":      ROUND_TIMEOUT,
                **self._question_payload(room),
            },
            to=room["id"],
        )
        self._start_round_timer(room)

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
            log_activity(player["user_id"], "playground")

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

    # ── Matchmaking ────────────────────────────────────────────────────────

    def _find_match(self, incoming: Dict) -> Optional[Dict]:
        now = time.time()
        best, best_wait = None, -1
        for waiter in self.waiting_queue:
            if waiter["user_id"] == incoming["user_id"]:
                continue
            waited    = now - waiter["joined_at"]
            tolerance = float("inf") if waited >= 60 else (6 if waited >= 30 else 3)
            diff      = abs(waiter["progress"] - incoming["progress"])
            if diff <= tolerance and waited > best_wait:
                best, best_wait = waiter, waited
        return best

    def _start_match(self, player_one: Dict, player_two: Dict) -> None:
        room_id    = f"playground-{uuid4().hex}"
        questions  = self._pick_questions()
        first_q    = questions[0]
        first_opts = _build_options(first_q)

        room_state = {
            "id":               room_id,
            "players":          [player_one, player_two],
            "questions":        questions,
            "current_question": first_q,
            "current_options":  first_opts,
            "round":            1,
            "scores":           {player_one["sid"]: 0, player_two["sid"]: 0},
            "active":           True,
            # ── New fields for fair play ──────────────────────────────
            "answered_sids":    set(),     # sids that have used their one attempt this round
            "question_sent_at": time.time(),
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
                "timeout":      ROUND_TIMEOUT,
                "p1_progress":  player_one["progress"],
                "p2_progress":  player_two["progress"],
                **self._question_payload(room_state),
            },
            to=room_id,
        )
        self._start_round_timer(room_state)

    def _pick_questions(self) -> List[Dict]:
        pool = QUESTIONS.copy()
        random.shuffle(pool)
        return pool[:TOTAL_ROUNDS]

    def join_queue(self, sid: str, user_id: int, username: str, progress: int) -> None:
        with self._lock:
            if any(w["sid"] == sid for w in self.waiting_queue):
                emit("queue_status", {"message": "You are already waiting for an opponent."})
                return
            if self._find_room_by_sid(sid):
                emit("queue_status", {"message": "You are already in a match."})
                return

            incoming = {
                "sid":       sid,
                "user_id":   user_id,
                "username":  username,
                "progress":  progress,
                "joined_at": time.time(),
            }
            opponent = self._find_match(incoming)

            if opponent is None:
                self.waiting_queue.append(incoming)
                emit("queue_status", {
                    "message": f"Waiting for an opponent at your level ({progress} modules completed)..."
                })
                return

            self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != opponent["sid"]]
            self._start_match(opponent, incoming)

    # ── Answer submission ──────────────────────────────────────────────────

    def submit_answer(self, sid: str, label: str) -> None:
        with self._lock:
            room = self._find_room_by_sid(sid)
            if not room:
                emit("answer_result", {"correct": False, "message": "No active match found."}, to=sid)
                return
            if not room["active"]:
                emit("answer_result", {"correct": False, "message": "Match already finished."}, to=sid)
                return
            if not label:
                emit("answer_result", {"correct": False, "message": "No option selected."}, to=sid)
                return

            # ── One attempt per player per round ──────────────────────
            if sid in room["answered_sids"]:
                emit("answer_result", {"correct": False, "message": "You have already used your attempt this round."}, to=sid)
                return

            # Record this player's attempt immediately — no second chances
            room["answered_sids"].add(sid)

            correct = self._is_correct(room, label)

            if correct:
                # Winner — resolve round immediately
                emit("answer_result", {"correct": True}, to=sid)
                self._resolve_round(room, winner_sid=sid)
            else:
                # Wrong — lock this player out, tell them
                emit(
                    "answer_result",
                    {
                        "correct": False,
                        "locked":  True,       # client should lock options permanently
                        "message": "Wrong answer. Waiting for your opponent…",
                        "chosen":  label.strip().upper(),
                    },
                    to=sid,
                )
                # If both players have now answered and both were wrong → resolve now
                if len(room["answered_sids"]) == 2:
                    self._resolve_round(room, winner_sid=None, timed_out=False)

    # ── Disconnect ─────────────────────────────────────────────────────────

    def disconnect(self, sid: str) -> None:
        with self._lock:
            before = len(self.waiting_queue)
            self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != sid]
            if len(self.waiting_queue) < before:
                return

            room = self._find_room_by_sid(sid)
            if not room or not room["active"]:
                return

            room["active"] = False
            leaver    = next(p for p in room["players"] if p["sid"] == sid)
            remaining = next(p for p in room["players"] if p["sid"] != sid)

            self._award_points(remaining["user_id"], room["scores"][remaining["sid"]] + DISCONNECT_BONUS)
            log_activity(remaining["user_id"], "playground")
            earned = room["scores"][leaver["sid"]]
            if earned > 0:
                self._award_points(leaver["user_id"], earned)
                log_activity(leaver["user_id"], "playground")

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


@socketio.on("join_playground")
def handle_join_playground():
    if not current_user.is_authenticated:
        emit("queue_status", {"message": "Please log in to use the playground."})
        return
    progress = _completed_count(current_user.id)
    matchmaker.join_queue(request.sid, current_user.id, current_user.username, progress)


@socketio.on("submit_answer")
def handle_submit_answer(data):
    label = data.get("label", "") if isinstance(data, dict) else ""
    matchmaker.submit_answer(request.sid, label)


@socketio.on("disconnect")
def handle_disconnect():
    matchmaker.disconnect(request.sid)
