from flask import render_template
from flask_login import current_user
from neurix.playground import playground
from neurix.models import ModuleProgress
from neurix.playground.data import MODULE_TOPICS


@playground.route('/playground')
def index():
    completed_modules = []
    completed_count   = 0

    if current_user.is_authenticated:
        rows = ModuleProgress.query.filter_by(
            user_id=current_user.id, completed=True
        ).all()
        completed_count   = len(rows)
        completed_modules = [
            {
                "id":    r.module_id,
                "title": MODULE_TOPICS.get(r.module_id, {}).get("title", r.module_id),
            }
            for r in rows
            if r.module_id in MODULE_TOPICS
        ]

    return render_template(
        'playground.html',
        title='Playground',
        completed_count=completed_count,
        completed_modules=completed_modules,
    )
