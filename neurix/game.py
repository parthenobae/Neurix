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
]

WIN_POINTS = 10


class Matchmaker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.waiting_player: Optional[Dict[str, str]] = None
        self.rooms: Dict[str, Dict] = {}

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _new_question(self) -> Dict[str, str]:
        return random.choice(QUESTIONS)

    def _is_correct(self, room: Dict, answer: str) -> bool:
        expected = room["question"]
        normalized = self._normalize(answer)
        valid_answers = [self._normalize(expected["answer"])]
        for alias in expected.get("aliases", []):
            valid_answers.append(self._normalize(alias))
        return normalized in valid_answers

    def _award_points(self, user_id: int) -> None:
        user = User.query.get(user_id)
        if not user:
            return
        try:
            user.points += WIN_POINTS
            db.session.commit()
        except Exception:
            db.session.rollback()

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
            question = self._new_question()
            room_state = {
                "id": room_id,
                "players": [player_one, player_two],
                "question": question,
                "winner_sid": None,
                "active": True,
            }
            self.rooms[room_id] = room_state

            join_room(room_id, sid=player_one["sid"])
            join_room(room_id, sid=player_two["sid"])

            emit(
                "match_found",
                {
                    "room_id": room_id,
                    "question": question["question"],
                    "players": [player_one["username"], player_two["username"]],
                    "points_to_win": WIN_POINTS,
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

            if room["winner_sid"] is not None:
                emit("answer_result", {"correct": False, "message": "A winner has already been declared."})
                return

            if self._is_correct(room, cleaned_answer):
                room["winner_sid"] = sid
                room["active"] = False
                winner = next(player for player in room["players"] if player["sid"] == sid)
                self._award_points(winner["user_id"])
                emit(
                    "match_ended",
                    {
                        "winner": winner["username"],
                        "reason": "correct_answer",
                        "points_awarded": WIN_POINTS,
                    },
                    to=room["id"],
                )
                room_id = room["id"]
                self.rooms.pop(room_id, None)
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
            loser = next(player for player in room["players"] if player["sid"] == sid)
            winner = next(player for player in room["players"] if player["sid"] != sid)
            room["winner_sid"] = winner["sid"]
            self._award_points(winner["user_id"])
            leave_room(room["id"], sid=loser["sid"])
            emit(
                "match_ended",
                {
                    "winner": winner["username"],
                    "reason": "opponent_disconnected",
                    "points_awarded": WIN_POINTS,
                },
                to=winner["sid"],
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
