from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from flask_socketio import SocketIO
from neurix.config import Config
from flask_migrate import Migrate


db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'
mail = Mail()
socketio = SocketIO(async_mode="eventlet")


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    socketio.init_app(app)
    migrate.init_app(app, db)

    from neurix.users.routes import users
    from neurix.posts.routes import posts
    from neurix.main.routes import main
    from neurix.playground import playground
    from neurix.learn import learn
    from neurix.datalab import datalab          # ← new
    from neurix.calendar import calendar          # ← add this

    app.register_blueprint(calendar, url_prefix='/calendar')  # ← add this
    app.register_blueprint(playground)
    app.register_blueprint(users)
    app.register_blueprint(posts)
    app.register_blueprint(main)
    app.register_blueprint(learn)
    app.register_blueprint(datalab)             # ← new
    
        # ── Global leaderboard context processor ─────────────────────────────────
    # Injects top-10 leaderboard into every template automatically.
    # Ordered by points desc, then username asc for tiebreaking.
    @app.context_processor
    def inject_leaderboard():
        from neurix.models import User
        top_users = (
            User.query
            .order_by(User.points.desc(), User.username.asc())
            .limit(10)
            .all()
        )
        return {
            "leaderboard": [
                {"username": u.username, "points": u.points}
                for u in top_users
            ]
        }

    return app
