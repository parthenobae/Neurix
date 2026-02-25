# Wiring log_activity into your other blueprints

Add this import at the top of each file listed below:

```python
from neurix.users.streak_utils import log_activity
```

---

## neurix/posts/routes.py
Inside `new_post()`, after `db.session.commit()`:
```python
log_activity(current_user.id, 'post')
```
-- done

---

## neurix/learn/routes.py

### When MCQ answer marks a module complete
Inside `complete_module()`, inside the `if not already_done:` block:
```python
log_activity(current_user.id, 'module')
```

### When code challenge marks a module complete
Inside `run_code()`, inside `if passed:` → `if not prog.completed:` block:
```python
log_activity(current_user.id, 'module')
```

### When unlock quiz is passed
Inside `submit_unlock_quiz()`, inside `if mcq_passed and code_passed:` → `if not existing:` block:
```python
log_activity(current_user.id, 'quiz')
```

---

## neurix/playground/game.py
Inside whatever function awards points after a match ends:
```python
log_activity(user_id, 'playground')
```

---

## Run db.create_all() once to create the new table

```python
from neurix import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
```
