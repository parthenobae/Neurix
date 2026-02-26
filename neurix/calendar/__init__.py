from flask import Blueprint

calendar = Blueprint('calendar', __name__)

from neurix.calendar import routes  # noqa: E402, F401