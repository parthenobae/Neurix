from flask import Blueprint

playground = Blueprint('playground', __name__)

from neurix.playground import routes, game  # noqa: F401, E402
