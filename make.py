from app import app, db

# Use the app context
with app.app_context():
    # Create the tables
    db.create_all()
