# THIS MUST BE THE ABSOLUTE FIRST LINE
import eventlet
eventlet.monkey_patch()

# Standard library imports
import os

# Your application imports
from neurix import create_app, socketio, db
from sqlalchemy import inspect

app = create_app()

def check_database():
    """Just verify database connection - tables already exist"""
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📊 Database connected! Found {len(tables)} tables")
            print("✅ Database is ready")
        except Exception as e:
            print(f"❌ Database connection error: {e}")

# Check database on startup (no migrations needed)
if os.environ.get('RENDER') or os.environ.get('DATABASE_URL'):
    check_database()

if __name__ == '__main__':
    # Get port from environment variable
    port = int(os.environ.get('PORT', 10000))
    
    # Bind to 0.0.0.0 to accept connections from anywhere
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
