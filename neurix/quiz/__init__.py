from flask import Blueprint

quiz = Blueprint('quiz', __name__)

from neurix.quiz import routes  # noqa: F401, E402
