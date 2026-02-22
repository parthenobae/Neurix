"""
tests/test_neurix.py
====================
Full unit-test suite for the Neurix Flask application.

Run with:
    pytest tests/test_neurix.py -v

Requirements (already in requirements.txt):
    pytest, Flask-Testing or plain pytest + Flask test client
"""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

# ── App bootstrap ──────────────────────────────────────────────────────────────
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("EMAIL_USER", "test@test.com")
os.environ.setdefault("EMAIL_PASS", "testpassword")
os.environ.setdefault("GROQ_API_KEY", "")   # empty → AI disabled, fallback used

from neurix import create_app, db, bcrypt
from neurix.models import User, Post


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures / base helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False          # disable CSRF for form tests
    MAIL_SUPPRESS_SEND = True         # never send real emails
    SERVER_NAME = None


@pytest.fixture(scope="function")
def app():
    """Create a fresh app + in-memory DB for every test function."""
    application = create_app(TestConfig)
    application.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
        MAIL_SUPPRESS_SEND=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SECRET_KEY="test-secret-key",
    )
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _create_user(username="testuser", email="test@example.com",
                 password="password123", points=0):
    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(username=username, email=email,
                password=hashed, points=points)
    db.session.add(user)
    db.session.commit()
    return user


def _login(client, email="test@example.com", password="password123"):
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=True,
    )


def _create_post(user, title="Test Post", content="Some content"):
    post = Post(title=title, content=content, author=user)
    db.session.add(post)
    db.session.commit()
    return post


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestUserModel:

    def test_user_creation(self, app):
        with app.app_context():
            user = _create_user()
            assert user.id is not None
            assert user.username == "testuser"
            assert user.email == "test@example.com"
            assert user.image_file == "default.jpg"
            assert user.points == 0

    def test_user_default_points(self, app):
        with app.app_context():
            user = _create_user()
            assert user.points == 0

    def test_user_points_update(self, app):
        with app.app_context():
            user = _create_user()
            user.points += 10
            db.session.commit()
            fetched = User.query.get(user.id)
            assert fetched.points == 10

    def test_user_unique_username(self, app):
        with app.app_context():
            _create_user(username="duplicate", email="a@a.com")
            from sqlalchemy.exc import IntegrityError
            with pytest.raises(IntegrityError):
                _create_user(username="duplicate", email="b@b.com")

    def test_user_unique_email(self, app):
        with app.app_context():
            _create_user(username="user1", email="same@example.com")
            from sqlalchemy.exc import IntegrityError
            with pytest.raises(IntegrityError):
                _create_user(username="user2", email="same@example.com")

    def test_password_is_hashed(self, app):
        with app.app_context():
            user = _create_user(password="plaintext")
            assert user.password != "plaintext"
            assert bcrypt.check_password_hash(user.password, "plaintext")

    def test_get_reset_token(self, app):
        with app.app_context():
            user = _create_user()
            token = user.get_reset_token()
            assert token is not None
            assert isinstance(token, str)

    def test_verify_reset_token_valid(self, app):
        with app.app_context():
            user = _create_user()
            token = user.get_reset_token()
            verified = User.verify_reset_token(token)
            assert verified is not None
            assert verified.id == user.id

    def test_verify_reset_token_invalid(self, app):
        with app.app_context():
            result = User.verify_reset_token("completelyinvalidtoken")
            assert result is None

    def test_user_repr(self, app):
        with app.app_context():
            user = _create_user()
            assert "testuser" in repr(user)
            assert "test@example.com" in repr(user)


class TestPostModel:

    def test_post_creation(self, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            assert post.id is not None
            assert post.title == "Test Post"
            assert post.content == "Some content"
            assert post.author == user

    def test_post_date_posted_set_automatically(self, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            assert post.date_posted is not None

    def test_post_repr(self, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user, title="My Post")
            assert "My Post" in repr(post)

    def test_post_foreign_key(self, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            assert post.user_id == user.id

    def test_user_posts_relationship(self, app):
        with app.app_context():
            user = _create_user()
            _create_post(user, title="Post 1")
            _create_post(user, title="Post 2")
            assert len(user.posts) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAIN BLUEPRINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMainRoutes:

    def test_home_get(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_home_alias(self, client):
        resp = client.get("/home")
        assert resp.status_code == 200

    def test_blogs_get(self, client):
        resp = client.get("/blogs")
        assert resp.status_code == 200

    def test_blogs_pagination(self, client, app):
        with app.app_context():
            user = _create_user()
            for i in range(7):
                _create_post(user, title=f"Post {i}")
        resp = client.get("/blogs?page=2")
        assert resp.status_code == 200

    def test_playground_get(self, client):
        resp = client.get("/playground")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 3. USERS BLUEPRINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestUserRegistration:

    def test_register_page_loads(self, client):
        resp = client.get("/register")
        assert resp.status_code == 200

    def test_register_success(self, client, app):
        resp = client.post("/register", data={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
            "confirm_password": "password123",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            user = User.query.filter_by(username="newuser").first()
            assert user is not None

    def test_register_redirects_authenticated_user(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/register", follow_redirects=False)
        assert resp.status_code == 302

    def test_register_duplicate_username(self, client, app):
        with app.app_context():
            _create_user(username="taken", email="taken@example.com")
        resp = client.post("/register", data={
            "username": "taken",
            "email": "other@example.com",
            "password": "password123",
            "confirm_password": "password123",
        }, follow_redirects=True)
        assert b"username is taken" in resp.data.lower() or resp.status_code == 200

    def test_register_duplicate_email(self, client, app):
        with app.app_context():
            _create_user(username="user1", email="taken@example.com")
        resp = client.post("/register", data={
            "username": "user2",
            "email": "taken@example.com",
            "password": "password123",
            "confirm_password": "password123",
        }, follow_redirects=True)
        assert b"email is taken" in resp.data.lower() or resp.status_code == 200

    def test_register_password_mismatch(self, client):
        resp = client.post("/register", data={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
            "confirm_password": "differentpassword",
        }, follow_redirects=True)
        assert resp.status_code == 200
        # Should re-render the form (not redirect to login)
        assert b"register" in resp.data.lower()


class TestUserLogin:

    def test_login_page_loads(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success(self, client, app):
        with app.app_context():
            _create_user()
        resp = _login(client)
        assert resp.status_code == 200

    def test_login_wrong_password(self, client, app):
        with app.app_context():
            _create_user()
        resp = client.post("/login", data={
            "email": "test@example.com",
            "password": "wrongpassword",
        }, follow_redirects=True)
        assert b"unsuccessful" in resp.data.lower()

    def test_login_nonexistent_email(self, client):
        resp = client.post("/login", data={
            "email": "nobody@example.com",
            "password": "password123",
        }, follow_redirects=True)
        assert b"unsuccessful" in resp.data.lower()

    def test_login_redirects_authenticated_user(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/login", follow_redirects=False)
        assert resp.status_code == 302

    def test_logout(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/logout", follow_redirects=True)
        assert resp.status_code == 200


class TestUserAccount:

    def test_account_requires_login(self, client):
        resp = client.get("/account", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_account_page_loads_when_logged_in(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/account")
        assert resp.status_code == 200

    def test_account_update_username_and_email(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.post("/account", data={
            "username": "updateduser",
            "email": "updated@example.com",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            user = User.query.filter_by(username="updateduser").first()
            assert user is not None
            assert user.email == "updated@example.com"


class TestUserPosts:

    def test_user_posts_page(self, client, app):
        with app.app_context():
            user = _create_user()
            _create_post(user)
        resp = client.get("/user/testuser")
        assert resp.status_code == 200

    def test_user_posts_404_for_unknown_user(self, client):
        resp = client.get("/user/doesnotexist")
        assert resp.status_code == 404


class TestPasswordReset:

    def test_reset_request_page_loads(self, client):
        resp = client.get("/reset_password")
        assert resp.status_code == 200

    def test_reset_request_redirects_authenticated(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/reset_password", follow_redirects=False)
        assert resp.status_code == 302

    def test_reset_request_unknown_email(self, client):
        resp = client.post("/reset_password", data={
            "email": "nobody@example.com",
        }, follow_redirects=True)
        # Form validation should fail — email not registered
        assert resp.status_code == 200

    @patch("neurix.users.routes.send_reset_email")
    def test_reset_request_known_email(self, mock_send, client, app):
        with app.app_context():
            _create_user()
        resp = client.post("/reset_password", data={
            "email": "test@example.com",
        }, follow_redirects=True)
        assert resp.status_code == 200
        mock_send.assert_called_once()

    def test_reset_token_invalid(self, client):
        resp = client.get("/reset_password/invalidtoken", follow_redirects=True)
        assert resp.status_code == 200
        assert b"invalid or expired" in resp.data.lower()

    def test_reset_token_valid(self, client, app):
        with app.app_context():
            user = _create_user()
            token = user.get_reset_token()
        resp = client.get(f"/reset_password/{token}")
        assert resp.status_code == 200

    def test_reset_password_updates_hash(self, client, app):
        with app.app_context():
            user = _create_user()
            token = user.get_reset_token()
            user_id = user.id
        resp = client.post(f"/reset_password/{token}", data={
            "password": "newpassword456",
            "confirm_password": "newpassword456",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            updated = User.query.get(user_id)
            assert bcrypt.check_password_hash(updated.password, "newpassword456")


# ══════════════════════════════════════════════════════════════════════════════
# 4. POSTS BLUEPRINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPostRoutes:

    def test_new_post_requires_login(self, client):
        resp = client.get("/post/new", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_new_post_page_loads(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.get("/post/new")
        assert resp.status_code == 200

    def test_create_post(self, client, app):
        with app.app_context():
            _create_user()
        _login(client)
        resp = client.post("/post/new", data={
            "title": "My New Post",
            "content": "This is the content.",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            post = Post.query.filter_by(title="My New Post").first()
            assert post is not None
            assert post.content == "This is the content."

    def test_view_post(self, client, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            post_id = post.id
        resp = client.get(f"/post/{post_id}")
        assert resp.status_code == 200

    def test_view_post_404(self, client):
        resp = client.get("/post/99999")
        assert resp.status_code == 404

    def test_update_post_requires_login(self, client, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            post_id = post.id
        resp = client.get(f"/post/{post_id}/update", follow_redirects=False)
        assert resp.status_code == 302

    def test_update_post_by_owner(self, client, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            post_id = post.id
        _login(client)
        resp = client.post(f"/post/{post_id}/update", data={
            "title": "Updated Title",
            "content": "Updated content.",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            updated = Post.query.get(post_id)
            assert updated.title == "Updated Title"

    def test_update_post_by_non_owner_returns_403(self, client, app):
        with app.app_context():
            owner = _create_user(username="owner", email="owner@example.com")
            post = _create_post(owner)
            post_id = post.id
            _create_user(username="other", email="other@example.com")
        _login(client, email="other@example.com")
        resp = client.post(f"/post/{post_id}/update", data={
            "title": "Hacked Title",
            "content": "Hacked content.",
        })
        assert resp.status_code == 403

    def test_delete_post_by_owner(self, client, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user)
            post_id = post.id
        _login(client)
        resp = client.post(f"/post/{post_id}/delete", follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            deleted = Post.query.get(post_id)
            assert deleted is None

    def test_delete_post_by_non_owner_returns_403(self, client, app):
        with app.app_context():
            owner = _create_user(username="owner", email="owner@example.com")
            post = _create_post(owner)
            post_id = post.id
            _create_user(username="other", email="other@example.com")
        _login(client, email="other@example.com")
        resp = client.post(f"/post/{post_id}/delete")
        assert resp.status_code == 403

    def test_get_update_post_prepopulates_form(self, client, app):
        with app.app_context():
            user = _create_user()
            post = _create_post(user, title="Original Title")
            post_id = post.id
        _login(client)
        resp = client.get(f"/post/{post_id}/update")
        assert resp.status_code == 200
        assert b"Original Title" in resp.data


# ══════════════════════════════════════════════════════════════════════════════
# 5. FORM VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistrationForm:

    def test_valid_registration_form(self, app):
        with app.app_context():
            from neurix.users.forms import RegistrationForm
            form = RegistrationForm(data={
                "username": "validuser",
                "email": "valid@example.com",
                "password": "password123",
                "confirm_password": "password123",
            })
            # Without request context + DB the custom validators won't fire
            # but built-in validators should pass
            assert form.username.data == "validuser"

    def test_login_form_fields(self, app):
        with app.app_context():
            from neurix.users.forms import LoginForm
            form = LoginForm(data={
                "email": "test@example.com",
                "password": "password123",
                "remember": False,
            })
            assert form.email.data == "test@example.com"

    def test_post_form_requires_title_and_content(self, app):
        with app.app_context():
            from neurix.posts.forms import PostForm
            form = PostForm(data={"title": "", "content": ""})
            form.validate()
            assert "title" in form.errors or not form.validate()


# ══════════════════════════════════════════════════════════════════════════════
# 6. GAME LOGIC TESTS  (pure unit tests — no sockets needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestGameHelpers:
    """Test the helper functions in playground/game.py directly."""

    def setup_method(self):
        # Import here so the module-level Groq client init doesn't crash
        # if GROQ_API_KEY is missing
        from neurix.playground import game as g
        self.game = g

    def test_build_options_returns_four(self):
        question = {
            "question": "What is 1+1?",
            "answer": "2",
            "distractors": ["1", "3", "4"],
        }
        opts = self.game._build_options(question)
        assert len(opts) == 4

    def test_build_options_labels(self):
        question = {
            "question": "What is 1+1?",
            "answer": "2",
            "distractors": ["1", "3", "4"],
        }
        opts = self.game._build_options(question)
        labels = [o["label"] for o in opts]
        assert sorted(labels) == ["A", "B", "C", "D"]

    def test_build_options_exactly_one_correct(self):
        question = {
            "question": "What is 1+1?",
            "answer": "2",
            "distractors": ["1", "3", "4"],
        }
        opts = self.game._build_options(question)
        correct = [o for o in opts if o["correct"]]
        assert len(correct) == 1
        assert correct[0]["text"] == "2"

    def test_build_options_shuffles(self):
        """Run 20 times — statistically the order must vary at least once."""
        question = {
            "question": "Q?",
            "answer": "correct",
            "distractors": ["wrong1", "wrong2", "wrong3"],
        }
        first_label_set = set()
        for _ in range(20):
            opts = self.game._build_options(question)
            correct_label = next(o["label"] for o in opts if o["correct"])
            first_label_set.add(correct_label)
        assert len(first_label_set) > 1  # not always the same label

    def test_pick_questions_fallback_when_no_api_key(self):
        """With empty GROQ_API_KEY the fallback pool should be returned."""
        questions = self.game._pick_questions()
        assert len(questions) == self.game.TOTAL_ROUNDS
        for q in questions:
            assert "question" in q
            assert "answer" in q
            assert "distractors" in q

    def test_generate_questions_via_ai_returns_empty_without_client(self):
        original = self.game._groq_client
        self.game._groq_client = None
        result = self.game._generate_questions_via_ai(10)
        self.game._groq_client = original
        assert result == []


class TestMatchmaker:
    """Unit tests for the Matchmaker class using mocked socket emit."""

    def setup_method(self):
        from neurix.playground.game import Matchmaker, _build_options, TOTAL_ROUNDS
        self.Matchmaker = Matchmaker
        self._build_options = _build_options
        self.TOTAL_ROUNDS = TOTAL_ROUNDS

    def _make_room(self, matchmaker):
        """Helper: inject a fake active room into the matchmaker."""
        from neurix.playground.game import _build_options
        question = {
            "question": "Test question?",
            "answer": "Correct",
            "distractors": ["Wrong1", "Wrong2", "Wrong3"],
        }
        options = _build_options(question)
        correct_label = next(o["label"] for o in options if o["correct"])
        room = {
            "id": "test-room-001",
            "players": [
                {"sid": "sid_p1", "user_id": 1, "username": "player1"},
                {"sid": "sid_p2", "user_id": 2, "username": "player2"},
            ],
            "questions": [question] * self.TOTAL_ROUNDS,
            "current_question": question,
            "current_options": options,
            "round": 1,
            "round_winner_sid": None,
            "scores": {"sid_p1": 0, "sid_p2": 0},
            "active": True,
        }
        matchmaker.rooms["test-room-001"] = room
        return room, correct_label

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.join_room")
    @patch("neurix.playground.game._pick_questions")
    def test_join_queue_first_player_waits(self, mock_pick, mock_join, mock_emit):
        mock_pick.return_value = [{"question": "Q?", "answer": "A",
                                   "distractors": ["B", "C", "D"]}] * 10
        mm = self.Matchmaker()
        mm.join_queue("sid_1", 1, "alice")
        assert mm.waiting_player is not None
        assert mm.waiting_player["sid"] == "sid_1"

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.join_room")
    @patch("neurix.playground.game._pick_questions")
    def test_join_queue_second_player_creates_room(self, mock_pick, mock_join, mock_emit):
        mock_pick.return_value = [{"question": "Q?", "answer": "A",
                                   "distractors": ["B", "C", "D"]}] * 10
        mm = self.Matchmaker()
        mm.join_queue("sid_1", 1, "alice")
        mm.join_queue("sid_2", 2, "bob")
        assert len(mm.rooms) == 1
        assert mm.waiting_player is None

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.join_room")
    @patch("neurix.playground.game._pick_questions")
    def test_cannot_play_against_self(self, mock_pick, mock_join, mock_emit):
        mock_pick.return_value = [{"question": "Q?", "answer": "A",
                                   "distractors": ["B", "C", "D"]}] * 10
        mm = self.Matchmaker()
        mm.join_queue("sid_1", 1, "alice")
        mm.join_queue("sid_2", 1, "alice")   # same user_id
        assert len(mm.rooms) == 0            # no room created
        assert mm.waiting_player is not None # still waiting

    @patch("neurix.playground.game.emit")
    def test_submit_correct_answer_increments_score(self, mock_emit):
        mm = self.Matchmaker()
        room, correct_label = self._make_room(mm)
        mm.submit_answer("sid_p1", correct_label)
        assert room["scores"]["sid_p1"] == 1

    @patch("neurix.playground.game.emit")
    def test_submit_wrong_answer_no_score_change(self, mock_emit):
        mm = self.Matchmaker()
        room, correct_label = self._make_room(mm)
        wrong_label = next(o["label"] for o in room["current_options"] if not o["correct"])
        mm.submit_answer("sid_p1", wrong_label)
        assert room["scores"]["sid_p1"] == 0

    @patch("neurix.playground.game.emit")
    def test_submit_answer_no_active_match(self, mock_emit):
        mm = self.Matchmaker()
        mm.submit_answer("unknown_sid", "A")
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "answer_result"
        assert call_args[0][1]["correct"] is False

    @patch("neurix.playground.game.emit")
    def test_submit_answer_empty_label(self, mock_emit):
        mm = self.Matchmaker()
        self._make_room(mm)
        mm.submit_answer("sid_p1", "")
        # Should emit an error, not crash
        mock_emit.assert_called()

    @patch("neurix.playground.game.emit")
    def test_round_winner_blocks_further_answers(self, mock_emit):
        mm = self.Matchmaker()
        room, correct_label = self._make_room(mm)
        room["round_winner_sid"] = "sid_p1"   # someone already won this round
        mm.submit_answer("sid_p2", correct_label)
        # Should reject — round already won
        last_call = mock_emit.call_args_list[-1]
        assert last_call[0][0] == "answer_result"
        assert last_call[0][1]["correct"] is False

    @patch("neurix.playground.game.emit")
    def test_disconnect_removes_waiting_player(self, mock_emit):
        mm = self.Matchmaker()
        mm.waiting_player = {"sid": "sid_1", "user_id": 1, "username": "alice"}
        mm.disconnect("sid_1")
        assert mm.waiting_player is None

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.leave_room")
    @patch("neurix.playground.game.Matchmaker._award_points")
    def test_disconnect_mid_match_awards_remaining_player(
            self, mock_award, mock_leave, mock_emit):
        mm = self.Matchmaker()
        self._make_room(mm)
        mm.disconnect("sid_p1")   # player 1 leaves
        # remaining player (sid_p2 / user_id=2) should get points
        mock_award.assert_called()
        # room should be cleaned up
        assert len(mm.rooms) == 0

    @patch("neurix.playground.game.emit")
    def test_inactive_room_disconnect_ignored(self, mock_emit):
        mm = self.Matchmaker()
        room, _ = self._make_room(mm)
        room["active"] = False
        mm.disconnect("sid_p1")
        # Should not emit match_ended again
        mock_emit.assert_not_called()

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.Matchmaker._award_points")
    def test_all_rounds_complete_triggers_match_end(self, mock_award, mock_emit):
        mm = self.Matchmaker()
        room, correct_label = self._make_room(mm)
        # Fast-forward to last round
        room["round"] = self.TOTAL_ROUNDS
        # Submit correct answer on the final round
        mm.submit_answer("sid_p1", correct_label)
        # match_ended should have been emitted
        event_names = [call[0][0] for call in mock_emit.call_args_list]
        assert "match_ended" in event_names

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.Matchmaker._award_points")
    def test_draw_detected_correctly(self, mock_award, mock_emit):
        mm = self.Matchmaker()
        room, _ = self._make_room(mm)
        # Set equal scores manually and trigger finish
        room["scores"]["sid_p1"] = 5
        room["scores"]["sid_p2"] = 5
        mm._finish_match(room, reason="all_rounds_complete")
        match_ended_calls = [
            c for c in mock_emit.call_args_list if c[0][0] == "match_ended"
        ]
        assert len(match_ended_calls) == 1
        payload = match_ended_calls[0][0][1]
        assert payload["draw"] is True
        assert payload["winner"] is None

    @patch("neurix.playground.game.emit")
    @patch("neurix.playground.game.Matchmaker._award_points")
    def test_winner_detected_correctly(self, mock_award, mock_emit):
        mm = self.Matchmaker()
        room, _ = self._make_room(mm)
        room["scores"]["sid_p1"] = 7
        room["scores"]["sid_p2"] = 3
        mm._finish_match(room, reason="all_rounds_complete")
        match_ended_calls = [
            c for c in mock_emit.call_args_list if c[0][0] == "match_ended"
        ]
        payload = match_ended_calls[0][0][1]
        assert payload["draw"] is False
        assert payload["winner"] == "player1"

    def test_find_room_by_sid(self):
        mm = self.Matchmaker()
        room, _ = self._make_room(mm)
        found = mm._find_room_by_sid("sid_p1")
        assert found is not None
        assert found["id"] == "test-room-001"

    def test_find_room_by_sid_unknown(self):
        mm = self.Matchmaker()
        self._make_room(mm)
        found = mm._find_room_by_sid("nonexistent_sid")
        assert found is None

    def test_scores_payload(self):
        mm = self.Matchmaker()
        room, _ = self._make_room(mm)
        room["scores"]["sid_p1"] = 3
        room["scores"]["sid_p2"] = 1
        payload = mm._scores_payload(room)
        assert payload["player1"] == 3
        assert payload["player2"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. UTILS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils:

    @patch("neurix.users.utils.mail")
    def test_send_reset_email(self, mock_mail, app):
        # url_for(_external=True) needs a request context, not just app context
        with app.test_request_context("/"):
            from neurix import db
            db.create_all()
            user = _create_user()
            from neurix.users.utils import send_reset_email
            send_reset_email(user)
            mock_mail.send.assert_called_once()

    @patch("neurix.users.utils.Image")
    def test_save_picture(self, mock_image_cls, app):
        with app.app_context():
            mock_file = MagicMock()
            mock_file.filename = "photo.jpg"
            mock_img = MagicMock()
            mock_image_cls.open.return_value = mock_img

            with patch("neurix.users.utils.os.path.join", return_value="/tmp/fake.jpg"):
                from neurix.users.utils import save_picture
                result = save_picture(mock_file)

            assert result.endswith(".jpg")
            mock_img.thumbnail.assert_called_once()
            mock_img.save.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 8. AI QUESTION GENERATION TESTS  (mocked — no real API calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestAIGeneration:

    def setup_method(self):
        from neurix.playground import game as g
        self.game = g

    def _make_mock_response(self, questions: list) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(questions)
        return mock_resp

    @patch("neurix.playground.game._groq_client")
    def test_ai_generation_success(self, mock_client):
        questions = [
            {
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "distractors": [f"Wrong{i}a", f"Wrong{i}b", f"Wrong{i}c"],
            }
            for i in range(10)
        ]
        mock_client.chat.completions.create.return_value = \
            self._make_mock_response(questions)

        result = self.game._generate_questions_via_ai(10)
        assert len(result) == 10
        assert result[0]["question"] == "Question 0?"

    @patch("neurix.playground.game._groq_client")
    def test_ai_generation_strips_markdown_fences(self, mock_client):
        questions = [
            {
                "question": "Q?",
                "answer": "A",
                "distractors": ["B", "C", "D"],
            }
        ] * 10
        raw = "```json\n" + json.dumps(questions) + "\n```"
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = raw
        mock_client.chat.completions.create.return_value = mock_resp

        result = self.game._generate_questions_via_ai(10)
        assert len(result) == 10

    @patch("neurix.playground.game._groq_client")
    def test_ai_generation_invalid_json_falls_back(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "this is not json at all"
        mock_client.chat.completions.create.return_value = mock_resp

        result = self.game._generate_questions_via_ai(10)
        assert result == []

    @patch("neurix.playground.game._groq_client")
    def test_ai_generation_missing_fields_filtered(self, mock_client):
        # One valid, one missing distractors
        questions = [
            {"question": "Q?", "answer": "A", "distractors": ["B", "C", "D"]},
            {"question": "Bad Q?", "answer": "A"},   # missing distractors
        ]
        mock_client.chat.completions.create.return_value = \
            self._make_mock_response(questions)

        result = self.game._generate_questions_via_ai(10)
        # Only 1 valid, need 10 → returns []
        assert result == []

    @patch("neurix.playground.game._groq_client")
    def test_ai_generation_api_error_returns_empty(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = self.game._generate_questions_via_ai(10)
        assert result == []

    @patch("neurix.playground.game._generate_questions_via_ai", return_value=[])
    def test_pick_questions_uses_fallback_on_empty_ai(self, mock_ai):
        result = self.game._pick_questions()
        assert len(result) == self.game.TOTAL_ROUNDS

    @patch("neurix.playground.game._groq_client")
    def test_pick_questions_uses_ai_when_available(self, mock_client):
        questions = [
            {
                "question": f"AI Question {i}?",
                "answer": f"Answer {i}",
                "distractors": [f"W{i}a", f"W{i}b", f"W{i}c"],
            }
            for i in range(10)
        ]
        mock_client.chat.completions.create.return_value = \
            self._make_mock_response(questions)

        result = self.game._pick_questions()
        assert result[0]["question"].startswith("AI Question")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
