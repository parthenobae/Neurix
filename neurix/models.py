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
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module_id   = db.Column(db.String(60), nullable=False)
    completed   = db.Column(db.Boolean, default=False, nullable=False)
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
