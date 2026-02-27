# THIS MUST BE THE ABSOLUTE FIRST LINE
import eventlet
eventlet.monkey_patch()

# Standard library imports
import os

# Your application imports
from neurix import create_app, socketio, db
from flask_migrate import upgrade, init, migrate  # Add all three
from sqlalchemy import inspect

app = create_app()

def setup_database():
    """Initialize and run migrations if needed"""
    with app.app_context():
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"📊 Found tables: {tables}")
        
        if not tables:  # No tables exist
            print("🔄 Database empty - setting up from scratch...")
            try:
                # Check if migrations folder exists
                if not os.path.exists('migrations'):
                    print("📁 Initializing migrations...")
                    init()
                    print("✅ Migrations initialized")
                
                print("📝 Creating initial migration...")
                migrate(message="Initial migration")
                print("✅ Migration created")
                
                print("🔄 Applying migrations...")
                upgrade()
                print("✅ Database setup complete!")
                
                # Verify tables were created
                inspector = inspect(db.engine)
                print(f"📊 Tables now: {inspector.get_table_names()}")
                
            except Exception as e:
                print(f"❌ Setup error: {e}")
                # Don't crash - let the app try to run anyway
        else:
            # Tables exist, just run any pending migrations
            print("🔄 Running any pending migrations...")
            try:
                upgrade()
                print("✅ Migrations complete!")
            except Exception as e:
                print(f"❌ Migration error: {e}")

# Run database setup on startup
if os.environ.get('RENDER') or os.environ.get('DATABASE_URL'):
    setup_database()

if __name__ == '__main__':
    socketio.run(app, debug=True)
