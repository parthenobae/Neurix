from flask import render_template
from neurix.visualizer import visualizer


@visualizer.route('/visualizer')
def index():
    return render_template('visualizer/index.html', title='Algorithm Visualizer')
