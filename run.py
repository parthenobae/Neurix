import eventlet
eventlet.monkey_patch()

import os
from neurix import create_app, socketio, db
from sqlalchemy import inspect


app = create_app()


def check_database():
    print("🔍 Checking database...")
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📊 Database connected! Found {len(tables)} tables: {tables[:5]}...")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
