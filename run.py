import eventlet
eventlet.monkey_patch()

from neurix import create_app, socketio
app = create_app()

if os.environ.get('RENDER') or os.environ.get('DATABASE_URL'):
    with app.app_context():
        try:
            print("🔄 Running database migrations...")
            upgrade()
            print("✅ Migrations complete!")
        except Exception as e:
            print(f"❌ Migration error: {e}")
            # Don't crash the app - let it try to run anyway
            # The error will be logged and you can debug

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=10000, debug=True)
