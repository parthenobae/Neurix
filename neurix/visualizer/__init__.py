from flask import Blueprint

visualizer = Blueprint('visualizer', __name__)

from neurix.visualizer import routes
