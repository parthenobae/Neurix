from __future__ import annotations

import random
import time
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from flask import request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room

from neurix import db, socketio
from neurix.models import User, ModuleProgress
from neurix.users.streak_utils import log_activity


# Each question has: question, answer (correct), distractors (3 wrong options)
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
LABELS = ["A", "B", "C", "D"]


def _build_options(question: Dict) -> List[Dict]:
    """Return shuffled list of {label, text, correct} for a question."""
    choices = [{"text": question["answer"], "correct": True}]
    for d in question["distractors"]:
        choices.append({"text": d, "correct": False})
    random.shuffle(choices)
    return [{"label": LABELS[i], "text": c["text"], "correct": c["correct"]}
            for i, c in enumerate(choices)]



def _completed_count(user_id: int) -> int:
    """Return number of modules this user has completed."""
    return ModuleProgress.query.filter_by(
        user_id=user_id, completed=True
    ).count()

class Matchmaker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.waiting_queue: List[Dict] = []   # each entry has sid, user_id, username, progress, joined_at
        self.rooms: Dict[str, Dict] = {}

    def _pick_questions(self) -> List[Dict]:
        pool = QUESTIONS.copy()
        random.shuffle(pool)
        return pool[:TOTAL_ROUNDS]

    def _award_points(self, user_id: int, amount: int) -> None:
        user = User.query.get(user_id)
        if not user:
            return
        try:
            user.points += amount
            db.session.commit()
            log_activity(user_id, 'playground')
        except Exception:
            db.session.rollback()

    def _scores_payload(self, room: Dict) -> Dict:
        return {p["username"]: room["scores"][p["sid"]] for p in room["players"]}

    def _question_payload(self, room: Dict) -> Dict:
        q    = room["current_question"]
        opts = room["current_options"]
        return {
            "question": q["question"],
            "options":  [{"label": o["label"], "text": o["text"]} for o in opts],
        }

    def _is_correct(self, room: Dict, label: str) -> bool:
        label = label.strip().upper()
        for opt in room["current_options"]:
            if opt["label"] == label and opt["correct"]:
                return True
        return False

    def _correct_label(self, room: Dict) -> str:
        for opt in room["current_options"]:
            if opt["correct"]:
                return opt["label"]
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

        if score_a > score_b:
            overall_winner = players[0]
        elif score_b > score_a:
            overall_winner = players[1]
        else:
            overall_winner = None

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

    def _find_match(self, incoming: Dict) -> Optional[Dict]:
        """
        Find the best waiting opponent for incoming player.
        Tolerance widens based on how long the incoming player would wait:
          0-30s  → ±3 modules
          30-60s → ±6 modules
          60s+   → any opponent
        We use the incoming player's own wait time (0 here since they just
        joined) but check how long each waiter has been waiting to decide
        tolerance for THEM — pick the waiter who has waited longest and
        is within tolerance.
        """
        import time
        now = time.time()
        best = None
        best_wait = -1

        for waiter in self.waiting_queue:
            if waiter["user_id"] == incoming["user_id"]:
                continue
            waited = now - waiter["joined_at"]
            if waited >= 60:
                tolerance = float("inf")
            elif waited >= 30:
                tolerance = 6
            else:
                tolerance = 3

            diff = abs(waiter["progress"] - incoming["progress"])
            if diff <= tolerance and waited > best_wait:
                best = waiter
                best_wait = waited

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
                "room_id":         room_id,
                "round":           1,
                "total_rounds":    TOTAL_ROUNDS,
                "players":         [player_one["username"], player_two["username"]],
                "scores":          {player_one["username"]: 0, player_two["username"]: 0},
                "question":        first_q["question"],
                "options":         [{"label": o["label"], "text": o["text"]} for o in first_opts],
                "p1_progress":     player_one["progress"],
                "p2_progress":     player_two["progress"],
            },
            to=room_id,
        )

    def join_queue(self, sid: str, user_id: int, username: str, progress: int) -> None:
        import time
        with self._lock:
            # Already waiting?
            if any(w["sid"] == sid for w in self.waiting_queue):
                emit("queue_status", {"message": "You are already waiting for an opponent."})
                return

            # Already in a match?
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
                # No suitable opponent yet — add to queue
                self.waiting_queue.append(incoming)
                emit("queue_status", {
                    "message": f"Waiting for an opponent at your level ({progress} modules completed)..."
                })
                return

            # Remove opponent from queue
            self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != opponent["sid"]]
            self._start_match(opponent, incoming)

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
            # Remove from queue if waiting
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
    label = ""
    if isinstance(data, dict):
        label = data.get("label", "")
    matchmaker.submit_answer(request.sid, label)


@socketio.on("disconnect")
def handle_disconnect():
    matchmaker.disconnect(request.sid)
