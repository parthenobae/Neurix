from __future__ import annotations

import random
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

from flask import request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room

from neurix import db, socketio
from neurix.models import User


QUESTIONS = [
    {
        "question": "Which algorithm is commonly used for binary classification and outputs probabilities using a sigmoid function?",
        "answer": "logistic regression",
    },
    {
        "question": "What does overfitting mean in machine learning?",
        "answer": "model learns noise",
        "aliases": [
            "memorizes training data",
            "poor generalization",
            "fits noise",
        ],
    },
    {
        "question": "Which validation strategy repeatedly splits data into training and validation folds?",
        "answer": "cross validation",
        "aliases": ["k fold cross validation", "k-fold cross validation", "k fold"],
    },
    {
        "question": "In gradient descent, what parameter controls the step size of each update?",
        "answer": "learning rate",
    },
    {
        "question": "Which metric is often preferred over accuracy for imbalanced binary datasets?",
        "answer": "f1 score",
        "aliases": ["f1", "precision recall", "auc pr"],
    },
    {
        "question": "What type of neural network layer connects every neuron to every neuron in the next layer?",
        "answer": "fully connected",
        "aliases": ["dense layer", "dense", "fully connected layer"],
    },
    {
        "question": "Which technique randomly drops neurons during training to reduce overfitting?",
        "answer": "dropout",
    },
    {
        "question": "What is the name of the process of adjusting model weights using the chain rule of calculus?",
        "answer": "backpropagation",
        "aliases": ["back propagation", "backprop"],
    },
    {
        "question": "Which unsupervised learning algorithm groups data points into k clusters?",
        "answer": "k means",
        "aliases": ["k-means", "k means clustering", "k-means clustering"],
    },
    {
        "question": "What do you call the set of hyperparameter values that minimizes validation loss?",
        "answer": "optimal hyperparameters",
        "aliases": ["best hyperparameters", "hyperparameter tuning", "tuned hyperparameters"],
    },
    {
        "question": "Which algorithm builds an ensemble of decision trees using random feature subsets?",
        "answer": "random forest",
    },
    {
        "question": "What term describes the error due to overly simple assumptions in a learning algorithm?",
        "answer": "bias",
        "aliases": ["high bias", "underfitting bias"],
    },
    {
        "question": "Which activation function outputs values strictly between 0 and 1 and is used in output layers for binary classification?",
        "answer": "sigmoid",
        "aliases": ["sigmoid function"],
    },
    {
        "question": "What is the name of the optimization algorithm that adapts learning rates for each parameter?",
        "answer": "adam",
        "aliases": ["adam optimizer"],
    },
    {
        "question": "Which dimensionality reduction technique projects data onto directions of maximum variance?",
        "answer": "pca",
        "aliases": ["principal component analysis"],
    },
]

TOTAL_ROUNDS = 10
POINTS_PER_ROUND = 1          # awarded to round winner immediately
DISCONNECT_BONUS = 3          # extra points given to remaining player on disconnect


class Matchmaker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.waiting_player: Optional[Dict[str, str]] = None
        self.rooms: Dict[str, Dict] = {}

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _pick_questions(self) -> list:
        pool = QUESTIONS.copy()
        random.shuffle(pool)
        return pool[:TOTAL_ROUNDS]

    def _is_correct(self, room: Dict, answer: str) -> bool:
        expected = room["current_question"]
        normalized = self._normalize(answer)
        valid_answers = [self._normalize(expected["answer"])]
        for alias in expected.get("aliases", []):
            valid_answers.append(self._normalize(alias))
        return normalized in valid_answers

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

    def _advance_round(self, room: Dict) -> None:
        """Move to the next round or end the match if all rounds are done."""
        room["round"] += 1
        room["round_winner_sid"] = None

        if room["round"] > TOTAL_ROUNDS:
            self._finish_match(room, reason="all_rounds_complete")
            return

        room["current_question"] = room["questions"][room["round"] - 1]
        emit(
            "next_round",
            {
                "round": room["round"],
                "total_rounds": TOTAL_ROUNDS,
                "question": room["current_question"]["question"],
                "scores": self._scores_payload(room),
            },
            to=room["id"],
        )

    def _finish_match(self, room: Dict, reason: str) -> None:
        room["active"] = False
        scores = room["scores"]
        players = room["players"]

        score_a = scores[players[0]["sid"]]
        score_b = scores[players[1]["sid"]]

        if score_a > score_b:
            overall_winner = players[0]
            overall_loser = players[1]
        elif score_b > score_a:
            overall_winner = players[1]
            overall_loser = players[0]
        else:
            overall_winner = None  # draw

        # Persist points equal to rounds won
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
                "draw": overall_winner is None,
            },
            to=room["id"],
        )
        self.rooms.pop(room["id"], None)

    def join_queue(self, sid: str, user_id: int, username: str) -> None:
        with self._lock:
            if self.waiting_player and self.waiting_player["sid"] == sid:
                emit("queue_status", {"message": "You are already waiting for an opponent."})
                return

            current_room = self._find_room_by_sid(sid)
            if current_room:
                emit("queue_status", {"message": "You are already in a match."})
                return

            if self.waiting_player is None:
                self.waiting_player = {"sid": sid, "user_id": user_id, "username": username}
                emit("queue_status", {"message": "Waiting for another player to join..."})
                return

            if self.waiting_player["user_id"] == user_id:
                emit(
                    "queue_status",
                    {"message": "You cannot play against yourself. Open the game from another account."},
                )
                return

            player_one = self.waiting_player
            player_two = {"sid": sid, "user_id": user_id, "username": username}
            self.waiting_player = None

            room_id = f"playground-{uuid4().hex}"
            questions = self._pick_questions()
            room_state = {
                "id": room_id,
                "players": [player_one, player_two],
                "questions": questions,
                "current_question": questions[0],
                "round": 1,
                "round_winner_sid": None,
                "scores": {player_one["sid"]: 0, player_two["sid"]: 0},
                "active": True,
            }
            self.rooms[room_id] = room_state

            join_room(room_id, sid=player_one["sid"])
            join_room(room_id, sid=player_two["sid"])

            emit(
                "match_found",
                {
                    "room_id": room_id,
                    "round": 1,
                    "total_rounds": TOTAL_ROUNDS,
                    "question": questions[0]["question"],
                    "players": [player_one["username"], player_two["username"]],
                    "scores": {player_one["username"]: 0, player_two["username"]: 0},
                },
                to=room_id,
            )

    def submit_answer(self, sid: str, answer: str) -> None:
        with self._lock:
            room = self._find_room_by_sid(sid)
            if not room:
                emit("answer_result", {"correct": False, "message": "No active match found."})
                return

            if not room["active"]:
                emit("answer_result", {"correct": False, "message": "Match already finished."})
                return

            cleaned_answer = answer.strip() if answer else ""
            if not cleaned_answer:
                emit("answer_result", {"correct": False, "message": "Answer cannot be empty."})
                return

            if room["round_winner_sid"] is not None:
                emit("answer_result", {"correct": False, "message": "Round already won. Next question incoming."})
                return

            if self._is_correct(room, cleaned_answer):
                room["round_winner_sid"] = sid
                room["scores"][sid] += POINTS_PER_ROUND
                winner = next(p for p in room["players"] if p["sid"] == sid)

                emit(
                    "round_ended",
                    {
                        "round": room["round"],
                        "round_winner": winner["username"],
                        "correct_answer": room["current_question"]["answer"],
                        "scores": self._scores_payload(room),
                    },
                    to=room["id"],
                )

                self._advance_round(room)
                return

            emit("answer_result", {"correct": False, "message": "Incorrect answer. Try again."}, to=sid)

    def disconnect(self, sid: str) -> None:
        with self._lock:
            if self.waiting_player and self.waiting_player["sid"] == sid:
                self.waiting_player = None
                return

            room = self._find_room_by_sid(sid)
            if not room or not room["active"]:
                return

            room["active"] = False
            leaver = next(p for p in room["players"] if p["sid"] == sid)
            remaining = next(p for p in room["players"] if p["sid"] != sid)

            # Give the remaining player a disconnect bonus on top of their current score
            self._award_points(remaining["user_id"], room["scores"][remaining["sid"]] + DISCONNECT_BONUS)
            # Still persist whatever the leaver had earned
            earned = room["scores"][leaver["sid"]]
            if earned > 0:
                self._award_points(leaver["user_id"], earned)

            leave_room(room["id"], sid=leaver["sid"])
            emit(
                "match_ended",
                {
                    "reason": "opponent_disconnected",
                    "scores": self._scores_payload(room),
                    "winner": remaining["username"],
                    "draw": False,
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
    matchmaker.join_queue(request.sid, current_user.id, current_user.username)


@socketio.on("submit_answer")
def handle_submit_answer(data):
    answer = ""
    if isinstance(data, dict):
        answer = data.get("answer", "")
    matchmaker.submit_answer(request.sid, answer)


@socketio.on("disconnect")
def handle_disconnect():
    matchmaker.disconnect(request.sid)
