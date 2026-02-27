import eventlet
eventlet.monkey_patch()

from neurix import create_app

app = create_app()
