import sqlite3
import hashlib
import re

def get_db():
    conn = sqlite3.connect("gym_app.db")
    conn.execute("PRAGMA foreign_keys = 1")
    return conn

def create_tables():
    conn = get_db()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )

    # Profiles table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER UNIQUE,
            age INTEGER,
            weight REAL,
            height REAL,
            username TEXT,
            allergies TEXT,
            training_type TEXT,
            diet_preferences TEXT,
            gender TEXT,
            goal TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    # Add columns if missing
    for col in ["username", "allergies", "training_type", "diet_preferences", "gender", "goal"]:
        try:
            cur.execute(f"ALTER TABLE profiles ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password_strength(password: str):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain at least one special character."
    return True, ""

def is_valid_email(email: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None

def register_user(email: str, password: str):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hash_password(password)),
        )
        user_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO profiles (
                user_id, age, weight, height, username,
                allergies, training_type, diet_preferences,
                gender, goal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, None, None, None, None, None, None, None, "Male", "Maintain"),
        )
        conn.commit()
        conn.close()
        return True, "Account created.", user_id
    except sqlite3.IntegrityError:
        conn.close()
        return False, "An account with this email already exists.", None

def verify_user(email: str, password: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    user_id, stored_hash = row
    if stored_hash == hash_password(password):
        return user_id
    return None

def reset_password(email: str, new_password: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return False, "No account found with this email."
    cur.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hash_password(new_password), email))
    conn.commit()
    conn.close()
    return True, "Password updated successfully."

def get_profile(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT age, weight, height,
               username, allergies, training_type, diet_preferences,
               gender, goal
        FROM profiles WHERE user_id = ?
        """,
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "age": row[0],
            "weight": row[1],
            "height": row[2],
            "username": row[3],
            "allergies": row[4],
            "training_type": row[5],
            "diet_preferences": row[6],
            "gender": row[7] or "Male",
            "goal": row[8] or "Maintain",
        }
    return {
        "age": None,
        "weight": None,
        "height": None,
        "username": None,
        "allergies": None,
        "training_type": None,
        "diet_preferences": None,
        "gender": "Male",
        "goal": "Maintain",
    }

def update_profile(user_id: int, age: int, weight: float, height: float,
                   username: str, allergies: str, training_type: str,
                   diet_preferences: str, gender: str, goal: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE profiles
        SET age = ?, weight = ?, height = ?,
            username = ?, allergies = ?,
            training_type = ?, diet_preferences = ?,
            gender = ?, goal = ?
        WHERE user_id = ?
        """,
        (age, weight, height, username, allergies, training_type, diet_preferences, gender, goal, user_id)
    )
    conn.commit()
    conn.close()
