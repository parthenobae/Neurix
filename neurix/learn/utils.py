
import os
import subprocess
import sys
import tempfile
from neurix.models import ModuleProgress, LevelUnlock, ChatMessage
from neurix import db


# ── Helpers ───────────────────────────────────────────────────────────────────
def _unlocked_levels(user_id: int) -> set:
    """Return set of level names the user has unlocked. Beginner always unlocked."""
    rows = LevelUnlock.query.filter_by(user_id=user_id).all()
    unlocked = {r.level for r in rows}
    unlocked.add("beginner")
    return unlocked


def _completed_module_ids(user_id: int) -> set:
    rows = ModuleProgress.query.filter_by(user_id=user_id, completed=True).all()
    return {r.module_id for r in rows}


def _get_or_create_progress(user_id: int, module_id: str) -> ModuleProgress:
    prog = ModuleProgress.query.filter_by(
        user_id=user_id, module_id=module_id
    ).first()
    if not prog:
        prog = ModuleProgress(user_id=user_id, module_id=module_id)
        db.session.add(prog)
        db.session.commit()
    return prog


def _execute_code(language: str, code: str) -> dict:
    """Execute code locally in a subprocess — no external API needed."""

    if language == "sql":
        code = """
import sqlite3, sys
conn = sqlite3.connect(':memory:')
cur = conn.cursor()
cur.executescript('''
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, years_experience INTEGER);
INSERT INTO employees VALUES (1,'Alice','Engineering',95000,5),(2,'Bob','Marketing',55000,3),(3,'Carol','Engineering',105000,8),(4,'Dave','HR',48000,2),(5,'Eve','Marketing',72000,6),(6,'Frank','Engineering',115000,10),(7,'Grace','HR',52000,4),(8,'Hank','Marketing',61000,5);
CREATE TABLE transactions (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, category TEXT);
INSERT INTO transactions VALUES (1,1,250.0,'Electronics'),(2,2,80.0,'Books'),(3,1,320.0,'Electronics');
CREATE TABLE customers (id INTEGER PRIMARY KEY, age INTEGER, income REAL, churn INTEGER, signup_date TEXT);
INSERT INTO customers VALUES (1,25,50000,0,'2022-06-01'),(2,35,75000,1,'2023-01-15');
''')
user_sql = \"\"\"""" + code.replace('"""', "'''") + """\"\"\"
try:
    cur.execute(user_sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    if cols:
        print(' | '.join(cols))
        print('-' * 40)
    for row in rows:
        print(' | '.join(str(v) for v in row))
except Exception as e:
    print(f'SQL Error: {e}', file=sys.stderr)
"""
        language = "python"

    lang_cmd = {
        "python":     [sys.executable],
        "javascript": ["node"],
    }

    if language not in lang_cmd:
        return {"stdout": "", "stderr": "", "error": f"Unsupported language: {language}"}

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py" if language == "python" else ".js",
            mode="w",
            delete=False,
        ) as f:
            f.write(code)
            fname = f.name

        result = subprocess.run(
            lang_cmd[language] + [fname],
            capture_output=True,
            text=True,
            timeout=5,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error":  result.stderr if result.returncode != 0 else "",
        }

    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "error": "Execution timed out (5s limit)."}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": "", "error": f"Runtime not found: {e}"}
    except Exception as exc:
        return {"stdout": "", "stderr": "", "error": f"Execution error: {exc}"}
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def _check_solution(output: str, checks: list) -> bool:
    """Return True if any check keyword appears in the output."""
    combined = output.lower()
    return any(str(c).lower() in combined for c in checks)

