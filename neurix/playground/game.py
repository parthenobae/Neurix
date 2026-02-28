"""
neurix/playground/game.py

Playground multiplayer quiz — rewritten with:
  1. Module-based matchmaking  — players are matched on COMMON completed modules,
                                  not just a raw count.  No common modules → no match.
  2. AI-generated questions    — Groq generates 10 questions per match from the
                                  topics both players have actually studied.
  3. Fallback safety           — if Groq fails, a clear error is sent rather than
                                  silently serving stale hardcoded questions.
"""
from __future__ import annotations

import json
import random
import time
import threading
from threading import Lock
from typing import Dict, List, Optional, Set
from uuid import uuid4

from flask import current_app, request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room
from groq import Groq

from neurix import db, socketio
from neurix.models import User, ModuleProgress
from neurix.users.streak_utils import log_activity
from neurix.playground.data import (
    TOTAL_ROUNDS, POINTS_PER_ROUND, DISCONNECT_BONUS,
    ROUND_TIMEOUT, LABELS, MODULE_TOPICS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _completed_module_ids(user_id: int) -> Set[str]:
    """Return the set of module_ids the user has completed."""
    rows = ModuleProgress.query.filter_by(user_id=user_id, completed=True).all()
    return {row.module_id for row in rows}


def _build_options(question: Dict) -> List[Dict]:
    """Shuffle answer + distractors and attach A/B/C/D labels."""
    choices = [{"text": question["answer"], "correct": True}]
    for d in question["distractors"]:
        choices.append({"text": d, "correct": False})
    random.shuffle(choices)
    return [
        {"label": LABELS[i], "text": c["text"], "correct": c["correct"]}
        for i, c in enumerate(choices)
    ]


# ── AI question generation ────────────────────────────────────────────────────

def _generate_questions_via_groq(common_module_ids: Set[str], app) -> List[Dict]:
    """
    Call Groq to generate TOTAL_ROUNDS questions covering the topics
    in common_module_ids.  Returns a list of dicts:
        [{"question": str, "answer": str, "distractors": [str, str, str]}, ...]

    Raises RuntimeError if generation fails.
    """
    # Build a readable topic list for the prompt
    topics = []
    for mid in common_module_ids:
        meta = MODULE_TOPICS.get(mid)
        if meta:
            topics.append(f"- {meta['title']}: {meta['description']}")

    if not topics:
        raise RuntimeError("No recognisable topics found for common modules.")

    topics_text = "\n".join(topics)

    prompt = f"""You are an expert ML educator. Generate exactly {TOTAL_ROUNDS} multiple-choice quiz questions
covering ONLY the following topics that both players have studied:

{topics_text}

STRICT RULES:
1. Each question must have exactly 1 correct answer and exactly 3 wrong distractors.
2. Questions must be factual and unambiguous.
3. Vary difficulty: mix easy recall and conceptual understanding questions.
4. Do NOT repeat the same concept twice.
5. Return ONLY a valid JSON array — no markdown, no explanation, no code fences.

Required JSON format (array of {TOTAL_ROUNDS} objects):
[
  {{
    "question": "Question text here?",
    "answer": "The correct answer",
    "distractors": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"]
  }}
]"""

    with app.app_context():
        api_key = app.config.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.6,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions = json.loads(raw)

    if not isinstance(questions, list) or len(questions) < TOTAL_ROUNDS:
        raise RuntimeError(
            f"Groq returned {len(questions)} questions, expected {TOTAL_ROUNDS}."
        )

    # Validate structure of each question
    validated = []
    for q in questions[:TOTAL_ROUNDS]:
        if (
            isinstance(q, dict)
            and isinstance(q.get("question"), str)
            and isinstance(q.get("answer"), str)
            and isinstance(q.get("distractors"), list)
            and len(q["distractors"]) == 3
        ):
            validated.append(q)

    if len(validated) < TOTAL_ROUNDS:
        raise RuntimeError(
            f"Only {len(validated)} questions passed validation."
        )

    return validated


# ── Matchmaker ────────────────────────────────────────────────────────────────

class Matchmaker:
    def __init__(self) -> None:
        self._lock         = Lock()
        self.waiting_queue: List[Dict] = []
        self.rooms:         Dict[str, Dict] = {}

    # ── Points & activity ─────────────────────────────────────────────────

    def _award_points(self, user_id: int, amount: int) -> None:
        user = db.session.get(User, user_id)
        if not user:
            return
        try:
            user.points += amount
            db.session.commit()
        except Exception:
            db.session.rollback()

    # ── Payload helpers ───────────────────────────────────────────────────

    def _scores_payload(self, room: Dict) -> Dict:
        return {p["username"]: room["scores"][p["sid"]] for p in room["players"]}

    def _question_payload(self, room: Dict) -> Dict:
        opts = room["current_options"]
        return {
            "question": room["current_question"]["question"],
            # correct field intentionally stripped — server-side only
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

    # ── Round timeout ─────────────────────────────────────────────────────

    def _start_round_timer(self, room: Dict) -> None:
        round_at_start = room["round"]
        room_id        = room["id"]

        def _expire():
            time.sleep(ROUND_TIMEOUT)
            with self._lock:
                r = self.rooms.get(room_id)
                if not r or not r["active"]:
                    return
                if r["round"] != round_at_start:
                    return
                self._resolve_round_from_thread(r)

        threading.Thread(target=_expire, daemon=True).start()

    # ── Thread-safe emitters (called from background threads) ─────────────

    def _resolve_round_from_thread(self, room: Dict) -> None:
        correct_lbl = self._correct_label(room)
        socketio.emit(
            "round_ended",
            {
                "round":          room["round"],
                "round_winner":   None,
                "correct_label":  correct_lbl,
                "correct_answer": room["current_question"]["answer"],
                "scores":         self._scores_payload(room),
                "timed_out":      True,
                "message":        "Time's up — no one answered correctly.",
            },
            to=room["id"],
        )
        self._advance_round_from_thread(room)

    def _advance_round_from_thread(self, room: Dict) -> None:
        room["round"]           += 1
        room["answered_sids"]    = set()
        room["question_sent_at"] = None

        if room["round"] > TOTAL_ROUNDS:
            self._finish_match_from_thread(room)
            return

        room["current_question"] = room["questions"][room["round"] - 1]
        room["current_options"]  = _build_options(room["current_question"])
        room["question_sent_at"] = time.time()

        socketio.emit(
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

    def _finish_match_from_thread(self, room: Dict) -> None:
        room["active"] = False
        self._finalise_match(room)
        socketio.emit(
            "match_ended",
            {
                "reason": "all_rounds_complete",
                "scores": self._scores_payload(room),
                **self._winner_payload(room),
            },
            to=room["id"],
        )
        self.rooms.pop(room["id"], None)

    # ── Core round logic ──────────────────────────────────────────────────

    def _resolve_round(self, room: Dict, winner_sid: Optional[str], timed_out: bool = False) -> None:
        correct_lbl = self._correct_label(room)

        if winner_sid:
            winner = next(p for p in room["players"] if p["sid"] == winner_sid)
            room["scores"][winner_sid] += POINTS_PER_ROUND
            winner_name = winner["username"]
            reason_msg  = f"{winner_name} answered correctly!"
        else:
            winner_name = None
            reason_msg  = (
                "Time's up — no one answered correctly."
                if timed_out
                else "Both players answered incorrectly."
            )

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
        room["round"]           += 1
        room["answered_sids"]    = set()
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

    def _winner_payload(self, room: Dict) -> Dict:
        scores  = room["scores"]
        players = room["players"]
        score_a = scores[players[0]["sid"]]
        score_b = scores[players[1]["sid"]]

        if   score_a > score_b: overall_winner = players[0]
        elif score_b > score_a: overall_winner = players[1]
        else:                   overall_winner = None

        return {
            "winner": overall_winner["username"] if overall_winner else None,
            "draw":   overall_winner is None,
        }

    def _finalise_match(self, room: Dict) -> None:
        """Award DB points and log activity for both players."""
        for player in room["players"]:
            pts = room["scores"][player["sid"]]
            if pts > 0:
                self._award_points(player["user_id"], pts)
            log_activity(player["user_id"], "playground")

    def _finish_match(self, room: Dict, reason: str) -> None:
        room["active"] = False
        self._finalise_match(room)
        emit(
            "match_ended",
            {
                "reason": reason,
                "scores": self._scores_payload(room),
                **self._winner_payload(room),
            },
            to=room["id"],
        )
        self.rooms.pop(room["id"], None)

    # ── Matchmaking ───────────────────────────────────────────────────────

    def _common_modules(self, modules_a: Set[str], modules_b: Set[str]) -> Set[str]:
        """Return module IDs both players have completed."""
        return modules_a & modules_b

    def _find_match(self, incoming: Dict) -> Optional[Dict]:
        """
        Find the best waiting opponent who shares at least one completed module.
        Among valid candidates, prefer the one waiting longest.
        After 60 s any opponent with ≥1 common module qualifies.
        After 120 s the min_common threshold drops to 1 regardless of level mix.
        """
        now = time.time()
        best, best_wait = None, -1

        for waiter in self.waiting_queue:
            # Never self-match
            if waiter["user_id"] == incoming["user_id"]:
                continue

            waited = now - waiter["joined_at"]

            common = self._common_modules(
                waiter["completed_modules"],
                incoming["completed_modules"],
            )

            # Must have at least 1 common module — hard requirement
            if not common:
                continue

            # Prefer matches with more common modules in the first 60 s
            min_common = 1 if waited >= 60 else 2

            if len(common) >= min_common and waited > best_wait:
                best, best_wait = waiter, waited

        return best

    def _start_match_async(self, player_one: Dict, player_two: Dict, app) -> None:
        """
        Kick off AI question generation in a background thread so the
        SocketIO event handler returns immediately.
        The thread emits 'match_found' once questions are ready, or
        'match_error' if Groq fails.
        """
        room_id = f"playground-{uuid4().hex}"

        # Put a placeholder in rooms so disconnect() can find these players
        placeholder = {
            "id":      room_id,
            "players": [player_one, player_two],
            "active":  False,   # not active until questions are ready
            "pending": True,
        }
        self.rooms[room_id] = placeholder

        join_room(room_id, sid=player_one["sid"])
        join_room(room_id, sid=player_two["sid"])

        # Tell both players that generation is in progress
        socketio.emit(
            "queue_status",
            {"message": "Generating questions with AI based on your shared topics…"},
            to=room_id,
        )

        common = self._common_modules(
            player_one["completed_modules"],
            player_two["completed_modules"],
        )

        def _generate_and_start():
            try:
                questions = _generate_questions_via_groq(common, app)
            except Exception as exc:
                # Groq failed — remove placeholder and inform players
                with self._lock:
                    self.rooms.pop(room_id, None)
                socketio.emit(
                    "match_error",
                    {"message": f"Could not generate questions: {exc}. Please try again."},
                    to=room_id,
                )
                return

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
                "pending":          False,
                "answered_sids":    set(),
                "question_sent_at": time.time(),
                "common_modules":   list(common),
            }

            with self._lock:
                self.rooms[room_id] = room_state

            # Build a readable topic list to show players what they're being tested on
            topic_titles = [
                MODULE_TOPICS[mid]["title"]
                for mid in common
                if mid in MODULE_TOPICS
            ]

            socketio.emit(
                "match_found",
                {
                    "room_id":        room_id,
                    "round":          1,
                    "total_rounds":   TOTAL_ROUNDS,
                    "players":        [player_one["username"], player_two["username"]],
                    "scores":         {player_one["username"]: 0, player_two["username"]: 0},
                    "timeout":        ROUND_TIMEOUT,
                    "common_topics":  topic_titles,
                    **self._question_payload(room_state),
                },
                to=room_id,
            )
            self._start_round_timer(room_state)

        threading.Thread(target=_generate_and_start, daemon=True).start()

    def join_queue(self, sid: str, user_id: int, username: str,
                   completed_modules: Set[str], app) -> None:
        with self._lock:
            # Guard: already queued?
            if any(w["sid"] == sid for w in self.waiting_queue):
                emit("queue_status", {"message": "You are already waiting for an opponent."})
                return

            # Guard: already in a match?
            if self._find_room_by_sid(sid):
                emit("queue_status", {"message": "You are already in a match."})
                return

            # Guard: no completed modules at all
            if not completed_modules:
                emit("queue_status", {
                    "message": "Complete at least one module before joining the playground."
                })
                return

            incoming = {
                "sid":               sid,
                "user_id":           user_id,
                "username":          username,
                "completed_modules": completed_modules,
                "joined_at":         time.time(),
            }

            opponent = self._find_match(incoming)

            if opponent is None:
                self.waiting_queue.append(incoming)
                module_count = len(completed_modules)
                emit("queue_status", {
                    "message": (
                        f"Searching for an opponent with overlapping modules "
                        f"(you've completed {module_count} module{'s' if module_count != 1 else ''})…"
                    )
                })
                # Start a 3-minute timeout — if no match found, tell the user
                self._start_queue_timeout(sid)
                return

            # Match found — remove opponent from queue and start async
            self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != opponent["sid"]]
            self._start_match_async(opponent, incoming, app)

    def _start_queue_timeout(self, sid: str) -> None:
        """
        After 3 minutes with no match, remove from queue and notify the player.
        """
        def _expire():
            time.sleep(180)
            with self._lock:
                before = len(self.waiting_queue)
                self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != sid]
                if len(self.waiting_queue) < before:
                    # Was still waiting — no match found
                    socketio.emit(
                        "queue_status",
                        {
                            "message": (
                                "No opponent with matching modules found. "
                                "Try again later or complete more modules to widen your match pool."
                            ),
                            "timed_out": True,
                        },
                        to=sid,
                    )

        threading.Thread(target=_expire, daemon=True).start()

    # ── Answer submission ─────────────────────────────────────────────────

    def submit_answer(self, sid: str, label: str) -> None:
        with self._lock:
            room = self._find_room_by_sid(sid)
            if not room:
                emit("answer_result", {"correct": False, "message": "No active match found."}, to=sid)
                return
            if room.get("pending"):
                emit("answer_result", {"correct": False, "message": "Match is still loading."}, to=sid)
                return
            if not room["active"]:
                emit("answer_result", {"correct": False, "message": "Match already finished."}, to=sid)
                return
            if not label:
                emit("answer_result", {"correct": False, "message": "No option selected."}, to=sid)
                return
            if sid in room["answered_sids"]:
                emit("answer_result", {"correct": False, "message": "You have already used your attempt this round."}, to=sid)
                return

            room["answered_sids"].add(sid)
            correct = self._is_correct(room, label)

            if correct:
                emit("answer_result", {"correct": True}, to=sid)
                self._resolve_round(room, winner_sid=sid)
            else:
                emit(
                    "answer_result",
                    {
                        "correct": False,
                        "locked":  True,
                        "message": "Wrong answer. Waiting for your opponent…",
                        "chosen":  label.strip().upper(),
                    },
                    to=sid,
                )
                if len(room["answered_sids"]) == 2:
                    self._resolve_round(room, winner_sid=None, timed_out=False)

    # ── Disconnect ────────────────────────────────────────────────────────

    def disconnect(self, sid: str) -> None:
        with self._lock:
            # Remove from queue if waiting
            before = len(self.waiting_queue)
            self.waiting_queue = [w for w in self.waiting_queue if w["sid"] != sid]
            if len(self.waiting_queue) < before:
                return

            room = self._find_room_by_sid(sid)
            if not room:
                return

            # Pending room (AI still generating) — just clean up
            if room.get("pending"):
                self.rooms.pop(room["id"], None)
                return

            if not room["active"]:
                return

            room["active"] = False
            leaver    = next(p for p in room["players"] if p["sid"] == sid)
            remaining = next(p for p in room["players"] if p["sid"] != sid)

            self._award_points(
                remaining["user_id"],
                room["scores"][remaining["sid"]] + DISCONNECT_BONUS,
            )
            log_activity(remaining["user_id"], "playground")

            earned = room["scores"][leaver["sid"]]
            if earned > 0:
                self._award_points(leaver["user_id"], earned)
                log_activity(leaver["user_id"], "playground")

            leave_room(room["id"], sid=leaver["sid"])
            socketio.emit(
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
            for player in room.get("players", []):
                if player["sid"] == sid:
                    return room
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

matchmaker = Matchmaker()


# ── SocketIO event handlers ───────────────────────────────────────────────────

@socketio.on("join_playground")
def handle_join_playground():
    if not current_user.is_authenticated:
        emit("queue_status", {"message": "Please log in to use the playground."})
        return

    app = current_app._get_current_object()
    completed = _completed_module_ids(current_user.id)
    matchmaker.join_queue(
        request.sid,
        current_user.id,
        current_user.username,
        completed,
        app,
    )


@socketio.on("submit_answer")
def handle_submit_answer(data):
    label = data.get("label", "") if isinstance(data, dict) else ""
    matchmaker.submit_answer(request.sid, label)


@socketio.on("disconnect")
def handle_disconnect():
    matchmaker.disconnect(request.sid)
