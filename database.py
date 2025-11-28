import sqlite3

DATABASE = "gym_app.db"

# -----------------------------------------
# Get DB connection
# -----------------------------------------
def get_connection():
    return sqlite3.connect(DATABASE, check_same_thread=False)

# -----------------------------------------
# CREATE TABLES  (same structure as app.py)
# -----------------------------------------
def create_tables():
    conn = get_connection()
    c = conn.cursor()

    # users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    # profiles table
    c.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            weight REAL,
            height REAL,
            diet_preferences TEXT,
            allergies TEXT,
            gender TEXT,
            training_type TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

# -----------------------------------------
# USER CREATION (register)
# -----------------------------------------
def create_user(username, email, password):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        """, (username, email, password))

        user_id = c.lastrowid

        c.execute("""
            INSERT INTO profiles (user_id, age, weight, height, diet_preferences, allergies, gender, training_type)
            VALUES (?, NULL, NULL, NULL, '', '', 'male', 'Kraft')
        """, (user_id,))

        conn.commit()
        return user_id
    except Exception as e:
        print("Error creating user:", e)
        return None
    finally:
        conn.close()

# -----------------------------------------
# LOGIN CHECK
# -----------------------------------------
def verify_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE email = ?", (email,))
    user = c.fetchone()

    conn.close()

    if user and user[1] == password:
        return user[0]
    return None

# -----------------------------------------
# GET PROFILE
# -----------------------------------------
def get_profile(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT age, weight, height, diet_preferences, allergies, gender, training_type
        FROM profiles
        WHERE user_id = ?
    """, (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "age": row[0],
        "weight": row[1],
        "height": row[2],
        "diet_preferences": row[3],
        "allergies": row[4],
        "gender": row[5],
        "training_type": row[6],
    }

# -----------------------------------------
# UPDATE PROFILE
# -----------------------------------------
def update_profile(user_id, age, weight, height, diet, allergies, gender, training_type):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        UPDATE profiles
        SET age = ?, weight = ?, height = ?, diet_preferences = ?,
            allergies = ?, gender = ?, training_type = ?
        WHERE user_id = ?
    """, (age, weight, height, diet, allergies, gender, training_type, user_id))

    conn.commit()
    conn.close()
    return True

# Run table creation on import
create_tables()
