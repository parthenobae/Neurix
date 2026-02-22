from flask import Blueprint

datalab = Blueprint('datalab', __name__)

from neurix.datalab import routes  # noqa: F401, E402
