from flask import render_template
from flask_login import login_required

from neurix.playground import playground


@playground.route('/playground')
def index():
    return render_template('playground.html', title='Playground')
