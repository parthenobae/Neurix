from flask import Blueprint

learn = Blueprint('learn', __name__)

from neurix.learn import routes  # noqa: F401, E402
