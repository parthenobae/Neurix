from flask import render_template
from flask_login import current_user
from neurix.playground import playground
from neurix.models import ModuleProgress


@playground.route('/playground')
def index():
    completed_count = 0
    if current_user.is_authenticated:
        completed_count = ModuleProgress.query.filter_by(
            user_id=current_user.id, completed=True
        ).count()

    return render_template(
        'playground.html',
        title='Playground',
        completed_count=completed_count,
    )
