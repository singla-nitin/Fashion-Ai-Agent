import os
import json
import uuid

DB_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(email, password):
    users = load_users()
    if any(u['email'] == email for u in users):
        return None  # Email already exists
    user_id = str(uuid.uuid4())
    users.append({'email': email, 'password': password, 'user_id': user_id})
    save_users(users)
    return user_id

def authenticate_user(email, password):
    users = load_users()
    for u in users:
        if u['email'] == email and u['password'] == password:
            return u['user_id']
    return None
