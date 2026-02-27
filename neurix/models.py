from datetime import datetime, timezone
from itsdangerous import URLSafeTimedSerializer as Serializer
from flask import current_app
from neurix import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)
    points = db.Column(db.Integer, nullable=False, default=0)
    module_progress = db.relationship('ModuleProgress', backref='user', lazy=True)
    level_unlocks = db.relationship('LevelUnlock', backref='user', lazy=True)
    activity_logs = db.relationship('ActivityLog', backref='user', lazy=True)
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True)
    calendar_tasks = db.relationship('CalendarTask', backref='user', lazy=True)
    user_roadmaps  = db.relationship('UserRoadmap',  backref='user', lazy=True)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token, max_age=expires_sec)['user_id']
        except Exception:
            return None
        return db.session.get(User, user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}', '{self.date_posted}')"


class ModuleProgress(db.Model):
    __tablename__ = 'module_progress'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module_id    = db.Column(db.String(60), nullable=False)
    completed    = db.Column(db.Boolean, default=False, nullable=False)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'module_id', name='uq_user_module'),
    )

    def mark_complete(self):
        self.completed = True
        self.completed_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"ModuleProgress(user={self.user_id}, module={self.module_id}, done={self.completed})"


class LevelUnlock(db.Model):
    __tablename__ = 'level_unlock'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    level       = db.Column(db.String(20), nullable=False)
    unlocked_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        db.UniqueConstraint('user_id', 'level', name='uq_user_level'),
    )

    def __repr__(self):
        return f"LevelUnlock(user={self.user_id}, level={self.level})"


class ActivityLog(db.Model):
    __tablename__ = 'activity_log'

    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date          = db.Column(db.Date, nullable=False)
    activity_type = db.Column(db.String(20), nullable=False)
    count         = db.Column(db.Integer, nullable=False, default=1)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'date', 'activity_type', name='uq_user_date_type'),
    )

    def __repr__(self):
        return f"ActivityLog(user={self.user_id}, date={self.date}, type={self.activity_type}, n={self.count})"


class ChatMessage(db.Model):
    __tablename__ = 'chat_message'

    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module_id = db.Column(db.String(60), nullable=False)
    role      = db.Column(db.String(10), nullable=False)
    content   = db.Column(db.Text, nullable=False)
    timestamp = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            "role":      self.role,
            "content":   self.content,
            "timestamp": self.timestamp.strftime("%H:%M"),
        }

    def __repr__(self):
        return f"ChatMessage(user={self.user_id}, module={self.module_id}, role={self.role})"


# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CalendarTask(db.Model):
    __tablename__ = 'calendar_task'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date            = db.Column(db.Date, nullable=False)
    title           = db.Column(db.String(200), nullable=False)
    description     = db.Column(db.Text, nullable=True)
    status          = db.Column(db.String(10), nullable=False, default='pending')
    source          = db.Column(db.String(10), nullable=False, default='manual')
    roadmap_task_id = db.Column(db.Integer, db.ForeignKey('roadmap_task.id'), nullable=True)
    created_at      = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            'id':          self.id,
            'date':        self.date.isoformat(),
            'title':       self.title,
            'description': self.description or '',
            'status':      self.status,
            'source':      self.source,
        }

    def __repr__(self):
        return f"CalendarTask(user={self.user_id}, date={self.date}, '{self.title}', {self.status})"


class Roadmap(db.Model):
    __tablename__ = 'roadmap'

    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    topic       = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    source      = db.Column(db.String(10), nullable=False, default='ai')
    owner_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at  = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    tasks       = db.relationship(
        'RoadmapTask', backref='roadmap', lazy=True,
        order_by='RoadmapTask.day_offset', cascade='all, delete-orphan'
    )
    enrollments = db.relationship(
        'UserRoadmap', backref='roadmap', lazy=True, cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f"Roadmap('{self.title}', source={self.source})"


class RoadmapTask(db.Model):
    __tablename__ = 'roadmap_task'

    id          = db.Column(db.Integer, primary_key=True)
    roadmap_id  = db.Column(db.Integer, db.ForeignKey('roadmap.id'), nullable=False)
    day_offset  = db.Column(db.Integer, nullable=False, default=0)
    title       = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)

    calendar_tasks = db.relationship('CalendarTask', backref='roadmap_task', lazy=True)

    def __repr__(self):
        return f"RoadmapTask(roadmap={self.roadmap_id}, day+{self.day_offset}, '{self.title}')"


class UserRoadmap(db.Model):
    __tablename__ = 'user_roadmap'

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    roadmap_id   = db.Column(db.Integer, db.ForeignKey('roadmap.id'), nullable=False)
    start_date   = db.Column(db.Date, nullable=False)
    tasks_seeded = db.Column(db.Boolean, nullable=False, default=False)
    enrolled_at  = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        db.UniqueConstraint('user_id', 'roadmap_id', name='uq_user_roadmap'),
    )

    def __repr__(self):
        return f"UserRoadmap(user={self.user_id}, roadmap={self.roadmap_id}, start={self.start_date})"


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ MODELS
# ─────────────────────────────────────────────────────────────────────────────

class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempt'

    id                   = db.Column(db.Integer, primary_key=True)
    user_id              = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tags                 = db.Column(db.String(500), nullable=False)
    score                = db.Column(db.Integer, nullable=False, default=0)
    total                = db.Column(db.Integer, nullable=False, default=15)
    time_taken           = db.Column(db.Integer, nullable=True)
    difficulty_breakdown = db.Column(db.Text, nullable=True)
    tag_breakdown        = db.Column(db.Text, nullable=True)
    attempted_at         = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"QuizAttempt(user={self.user_id}, score={self.score}/{self.total}, tags={self.tags})"