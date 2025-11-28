import streamlit as st
# Ganz am Anfang: breites Layout für die gesamte App
st.set_page_config(
    page_title="Fitness App",
    layout="wide",          # das sorgt dafür, dass alles breit angezeigt wird
    initial_sidebar_state="expanded"
)
import sqlite3
import hashlib
import re  # password + email checks
import pandas as pd  # demo chart on Progress page
import base64  # for background image + logo

import workout_planner  # teammates' workout builder
import workout_calendar  # teammates' calendar
import calorie_tracker   # ML-based calorie & protein tracker
import nutrition_advisory
import calories_nutrition
from nutrition_advisory import main as nutrition_main, load_and_prepare_data, DATA_URL

# DataFrame nur einmal beim App-Start laden
if "recipes_df" not in st.session_state:
    with st.spinner("Loading recipe data..."):
        st.session_state.recipes_df = load_and_prepare_data(DATA_URL)


# ---------- basic page setup ----------
st.set_page_config(
    page_title="UniFit Coach",
    page_icon="💪",
    layout="wide",
)

# ---------- colors ----------
PRIMARY_GREEN = "#007A3D"  # approx. HSG green


# ---------- helper: load image as base64 for login background ----------
def get_base64_of_image(path: str) -> str:
    """Read a local image file and return it as base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_logo(path: str) -> str:
    """Load logo and return as base64 string for embedding in HTML."""
    try:
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except FileNotFoundError:
        return ""


# image + logo files must exist in the same folder as app.py
BACKGROUND_IMAGE = get_base64_of_image("background_pitch.jpg")
LOGO_IMAGE = load_logo("unifit_logo.png")


# ---------- GLOBAL CSS (theme) ----------
st.markdown(
    f"""
    <style>
    /* main app container: max-width removed so pages can be full width */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
        margin: 0 auto !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    /* white background + green text for header bar */
    [data-testid="stHeader"] {{
        background-color: #FFFFFF !important;
        color: {PRIMARY_GREEN};
        box-shadow: none !important;
    }}

    /* generic buttons in main area (Login, Save profile, etc.) */
    .stButton > button {{
        border-radius: 999px;
        background-color: {PRIMARY_GREEN};
        color: #ffffff;
        border: 1px solid {PRIMARY_GREEN};
        padding: 0.5rem 1rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: #005c2d;
        border-color: #005c2d;
        color: #ffffff;
    }}

    /* sidebar background: light grey with subtle border */
    [data-testid="stSidebar"] {{
        background: #f5f7f6;
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }}

    /* default text elements in green */
    p, span, label, .stMarkdown, .stText, .stCaption {{
        color: {PRIMARY_GREEN};
    }}

    /* headings in HSG green */
    h1, h2, h3, h4 {{
        color: {PRIMARY_GREEN};
    }}

    /* rounded cards (containers with border=True) */
    div[data-testid="stVerticalBlock"] > div > div[style*="border-radius: 0.5rem"] {{
        border-radius: 1rem !important;
    }}

    /* ---- number inputs (Age, Weight, Height) styled like pills ---- */
    div[data-testid="stNumberInput"] input {{
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
        border-radius: 999px !important;
        border: 1px solid {PRIMARY_GREEN} !important;
        padding: 0.25rem 0.75rem !important;
    }}

    div[data-testid="stNumberInput"] input:focus {{
        outline: none !important;
        border: 2px solid {PRIMARY_GREEN} !important;
        box-shadow: 0 0 0 1px rgba(0, 122, 61, 0.25);
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
    }}

    div[data-testid="stNumberInput"] button {{
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
        border-radius: 999px !important;
        border: 1px solid {PRIMARY_GREEN} !important;
    }}

    div[data-testid="stNumberInput"] button:hover {{
        background-color: {PRIMARY_GREEN} !important;
        color: #ffffff !important;
        border-color: {PRIMARY_GREEN} !important;
    }}

    /* ---- text & password inputs (login / register / reset) ---- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPasswordInput"] input {{
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
        border-radius: 999px !important;
        border: 1px solid {PRIMARY_GREEN} !important;
        padding: 0.4rem 0.75rem !important;
    }}

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stPasswordInput"] input::placeholder {{
        color: rgba(0, 122, 61, 0.6) !important;
    }}

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stPasswordInput"] input:focus {{
        outline: none !important;
        border: 2px solid {PRIMARY_GREEN} !important;
        box-shadow: 0 0 0 1px rgba(0, 122, 61, 0.25);
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
    }}

    /* code blocks – white background instead of black */
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] {{
        background-color: #FFFFFF !important;
        color: {PRIMARY_GREEN} !important;
        border-radius: 0.75rem !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATABASE + SECURITY
# =========================================================

def get_db():
    """Open a connection to the SQLite database file."""
    conn = sqlite3.connect("gym_app.db")
    conn.execute("PRAGMA foreign_keys = 1")
    return conn


def create_tables():
    """Create users and profiles tables if they don't exist, add missing columns."""
    conn = get_db()
    cur = conn.cursor()

    # users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    # profiles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER UNIQUE,
            age INTEGER,
            weight REAL,
            height REAL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # add missing columns if not exist
    additional_cols = [
        "username TEXT",
        "allergies TEXT",
        "training_type TEXT",
        "diet_preferences TEXT",
        "gender TEXT DEFAULT 'Male'",
        "goal TEXT DEFAULT 'Maintain'"
    ]
    for col_def in additional_cols:
        try:
            cur.execute(f"ALTER TABLE profiles ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()
    conn.close()



def hash_password(password: str) -> str:
    """Hash a password string with SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ---------- password + email validation ----------

def validate_password_strength(password: str):
    """
    Check if password is strong enough:
    - at least 8 characters
    - at least one lowercase letter
    - at least one uppercase letter
    - at least one digit
    - at least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, (
            "Password must contain at least one special character "
            "(e.g. !, ?, #, ...)."
        )
    return True, ""


def is_valid_email(email: str) -> bool:
    """Simple email format validation."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None


# =========================================================
# AUTHENTICATION LOGIC
# =========================================================

def register_user(email: str, password: str):
    """Create a new user and an empty profile. Return (ok, msg, user_id)."""
    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hash_password(password)),
        )
        user_id = cur.lastrowid

        # create empty profile row for the new user
        cur.execute(
            """
            INSERT INTO profiles (
                user_id, age, weight, height,
                username, allergies, training_type, diet_preferences
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, None, None, None, None, None, None, None),
        )

        conn.commit()
        conn.close()
        return True, "Account created.", user_id
    except sqlite3.IntegrityError:
        conn.close()
        return False, "An account with this email already exists.", None


def verify_user(email: str, password: str):
    """Return user_id if email/password are correct, otherwise None."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, password_hash FROM users WHERE email = ?",
        (email,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    user_id, stored_hash = row
    if stored_hash == hash_password(password):
        return user_id
    return None


def reset_password(email: str, new_password: str):
    """Reset password for a given email (demo version: no email verification)."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return False, "No account found with this email."

    cur.execute(
        "UPDATE users SET password_hash = ? WHERE email = ?",
        (hash_password(new_password), email),
    )
    conn.commit()
    conn.close()
    return True, "Password updated successfully."


# =========================================================
# PROFILE DB ACCESS
# =========================================================

def get_profile(user_id: int):
    """Fetch profile info for a given user_id."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT age, weight, height,
               username, allergies, training_type, diet_preferences
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
        }

    return {
        "age": None,
        "weight": None,
        "height": None,
        "username": None,
        "allergies": None,
        "training_type": None,
        "diet_preferences": None,
    }


def update_profile(
    user_id: int,
    age: int,
    weight: float,
    height: float,
    username: str,
    allergies: str,
    training_type: str,
    diet_preferences: str,
):
    """Update profile values for a given user_id."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE profiles
        SET age = ?, weight = ?, height = ?,
            username = ?, allergies = ?,
            training_type = ?, diet_preferences = ?
        WHERE user_id = ?
        """,
        (
            age,
            weight,
            height,
            username,
            allergies,
            training_type,
            diet_preferences,
            user_id,
        ),
    )
    conn.commit()
    conn.close()


# =========================================================
# AUTHENTICATION UI (LOGIN / REGISTER / RESET)
# =========================================================

def show_login_page():
    """Login screen styled with centered card."""
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.title("Login")
        st.caption("Log in to your UniFit Coach dashboard.")

        with st.container(border=True):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    user_id = verify_user(email, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.user_email = email
                        st.session_state.current_page = "Profile"
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        st.write("---")
        st.write("Don't have an account yet?")
        if st.button("Create a new account", use_container_width=True):
            st.session_state.login_mode = "register"
            st.rerun()

        st.write("")
        if st.button("Forgot password?", use_container_width=True):
            st.session_state.login_mode = "reset"
            st.rerun()


def show_register_page():
    """Registration screen styled with centered card and password rules."""
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.title("Register")
        st.caption("Create an account for UniFit Coach.")

        with st.container(border=True):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            st.markdown(
                """
                **Password must contain:**
                - at least 8 characters  
                - at least one lowercase letter  
                - at least one uppercase letter  
                - at least one digit  
                - at least one special character (e.g. `!`, `?`, `#`, `@`)
                """,
                unsafe_allow_html=False,
            )

            if st.button("Register", use_container_width=True):
                if not email or not password:
                    st.error("Please enter both email and password.")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    ok_pw, msg_pw = validate_password_strength(password)
                    if not ok_pw:
                        st.error(msg_pw)
                    else:
                        ok, msg, user_id = register_user(email, password)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user_id
                            st.session_state.user_email = email
                            st.session_state.current_page = "Profile"
                            st.success("Account created! Let's set up your profile.")
                            st.rerun()
                        else:
                            st.error(msg)

        st.write("---")
        if st.button("Back to login", use_container_width=True):
            st.session_state.login_mode = "login"
            st.rerun()


def show_reset_password_page():
    """Simple password reset: enter email + new password (demo, no email verification)."""
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.title("Reset password")
        st.caption(
            "For demo purposes, you can reset your password by entering your email "
            "and a new password."
        )

        with st.container(border=True):
            email = st.text_input("Email")
            new_pw = st.text_input("New password", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")

            if st.button("Reset password", use_container_width=True):
                if not email or not new_pw or not confirm_pw:
                    st.error("Please fill out all fields.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    ok_pw, msg_pw = validate_password_strength(new_pw)
                    if not ok_pw:
                        st.error(msg_pw)
                    else:
                        ok, msg = reset_password(email, new_pw)
                        if ok:
                            st.success(msg)
                            st.session_state.login_mode = "login"
                            st.rerun()
                        else:
                            st.error(msg)

        st.write("---")
        if st.button("Back to login", use_container_width=True):
            st.session_state.login_mode = "login"
            st.rerun()


# =========================================================
# PUMPFESSOR JOE – SIMPLE IN-APP GUIDE
# =========================================================

def show_pumpfessor_joe(page_name: str):
    """Small helper box with tips depending on the current page."""
    with st.expander("👨‍🏫 Pumpfessor Joe – Need a quick guide?", expanded=False):
        if page_name == "Profile":
            st.write(
                "Welcome to your **Profile**! 🧍‍♂️\n\n"
                "- Enter your age, weight, height, preferences and allergies.\n"
                "- Click **Save profile**.\n"
                "- This data can be used to personalize your workouts "
                "and nutrition advice."
            )
        elif page_name == "Trainer":
            st.write(
                "This is the **Trainer** page. 🏋️‍♂️\n\n"
                "Use the tabs to build a workout with Pumpfessor Joe and "
                "see your long-term training schedule."
            )
        elif page_name == "Calorie tracker":
            st.write(
                "On the **Calorie tracker** page 🔥 you can:\n"
                "- Enter your body data and training session.\n"
                "- Let Pumpfessor Joe estimate your daily calorie target.\n"
                "- Log your meals and track calories & protein with donut charts."
            )
        elif page_name == "Nutrition adviser":
            st.write(
                "The **Nutrition adviser** page 🥗 will later give you suggestions on "
                "meals or macros based on your goals and allergies."
            )
        elif page_name == "Progress":
            st.write(
                "The **Progress** page 📈 shows how you're doing over time.\n\n"
                "Right now you see a demo chart. In the future, this can be replaced "
                "with real workout or calorie data."
            )
        else:
            st.write(
                "Pumpfessor Joe is here to help you navigate UniFit Coach. "
                "Use the menu on the left to switch between pages."
            )


# =========================================================
# APP PAGES
# =========================================================

# =========================================================
# PROFILE DB ACCESS (updated copy is below too)
# =========================================================

def get_profile(user_id: int):
    """Fetch profile info for a given user_id."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT age, weight, height,
               username, allergies, training_type, diet_preferences
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
        }

    return {
        "age": None,
        "weight": None,
        "height": None,
        "username": None,
        "allergies": None,
        "training_type": None,
        "diet_preferences": None,
    }


def update_profile(
    user_id: int,
    age: int,
    weight: float,
    height: float,
    username: str,
    allergies: str,
    training_type: str,
    diet_preferences: str,
):
    """Update profile values for a given user_id."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE profiles
        SET age = ?, weight = ?, height = ?,
            username = ?, allergies = ?,
            training_type = ?, diet_preferences = ?
        WHERE user_id = ?
        """,
        (
            age,
            weight,
            height,
            username,
            allergies,
            training_type,
            diet_preferences,
            user_id,
        ),
    )
    conn.commit()
    conn.close()


# =========================================================
# PROFILE DB ACCESS (updated with gender + goal) - KEEP this one too
# =========================================================

def get_profile(user_id: int):
    """Fetch profile info for a given user_id."""
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


def update_profile(
    user_id: int,
    age: int,
    weight: float,
    height: float,
    username: str,
    allergies: str,
    training_type: str,
    diet_preferences: str,
    gender: str,
    goal: str,
):
    """Update profile values for a given user_id."""
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
        (
            age,
            weight,
            height,
            username,
            allergies,
            training_type,
            diet_preferences,
            gender,
            goal,
            user_id,
        ),
    )
    conn.commit()
    conn.close()


# =========================================================
# PROFILE PAGE (updated)
# =========================================================

def show_profile_page():
    """Profile page with inputs stored in the database, including Gender + Goal."""
    user_id = st.session_state.user_id
    profile = get_profile(user_id)

    st.header("Profile")
    st.write("Basic information that can be used by the trainer and nutrition logic later.")
    st.divider()

    # Use a full-width container for the profile form so it can expand
    with st.container():
        with st.container(border=True):
            st.subheader("Your data")

            # keep two-column layout for form fields inside the full width container
            c1, c2 = st.columns(2)

            with c1:
                age = st.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=profile["age"] if profile["age"] is not None else 0,
                    step=1,
                )

                height = st.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=profile["height"] if profile["height"] is not None else 0.0,
                    step=0.5,
                )

                username = st.text_input(
                    "Username",
                    value=profile["username"] or "",
                    max_chars=30,
                )

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=0 if profile["gender"] == "Male" else 1,
                )

            with c2:
                weight = st.number_input(
                    "Weight (kg)",
                    min_value=0.0,
                    max_value=500.0,
                    value=profile["weight"] if profile["weight"] is not None else 0.0,
                    step=0.5,
                )

                training_options = [
                    "Not set",
                    "Strength",
                    "Hypertrophy",
                    "Endurance",
                    "Mixed",
                ]
                current_training = profile["training_type"] or "Not set"
                if current_training not in training_options:
                    current_training = "Not set"
                training_type = st.selectbox(
                    "Preferred training style",
                    training_options,
                    index=training_options.index(current_training),
                )

                diet_options = [
                    "Not set",
                    "No preference",
                    "High protein",
                    "Vegetarian",
                    "Vegan",
                    "Low carb",
                    "Mediterranean",
                ]
                current_diet = profile["diet_preferences"] or "Not set"
                if current_diet not in diet_options:
                    current_diet = "Not set"
                diet_preferences = st.selectbox(
                    "Diet preference",
                    diet_options,
                    index=diet_options.index(current_diet),
                )

                goal_options = ["Cut", "Maintain", "Bulk"]
                current_goal = profile["goal"] or "Maintain"
                if current_goal not in goal_options:
                    current_goal = "Maintain"
                goal = st.selectbox(
                    "Goal",
                    goal_options,
                    index=goal_options.index(current_goal),
                )

            allergies = st.text_area(
                "Allergies (optional)",
                value=profile["allergies"] or "",
                help="For example: peanuts, lactose, gluten …",
            )

            if st.button("Save profile", use_container_width=True):
                update_profile(
                    user_id,
                    int(age),
                    float(weight),
                    float(height),
                    username.strip() or None,
                    allergies.strip() or None,
                    training_type,
                    diet_preferences,
                    gender,
                    goal,
                )
                st.success("Profile saved.")

    st.divider()
    st.subheader("Current profile data")

    # reload profile from DB (in case it changed)
    profile = get_profile(user_id)

    st.write(f"**Username:** {profile['username'] or 'Not set'}")
    st.write(f"**Age:** {profile['age'] or 'Not set'} years")
    st.write(f"**Weight:** {profile['weight'] or 'Not set'} kg")
    st.write(f"**Height:** {profile['height'] or 'Not set'} cm")
    st.write(f"**Gender:** {profile['gender']}")
    st.write(f"**Goal:** {profile['goal']}")
    st.write(f"**Training style:** {profile['training_type'] or 'Not set'}")
    st.write(f"**Diet preference:** {profile['diet_preferences'] or 'Not set'}")
    st.write(f"**Allergies:** {profile['allergies'] or 'None noted'}")

    # --- Profile completeness indicator ---
    fields_for_completeness = [
        profile["username"],
        profile["age"],
        profile["weight"],
        profile["height"],
        profile["training_type"],
        profile["diet_preferences"],
        profile["gender"],
        profile["goal"],
    ]
    filled_fields = sum(
        1
        for v in fields_for_completeness
        if v not in (None, 0, 0.0, "", "Not set")
    )
    completeness = filled_fields / len(fields_for_completeness)
    st.write("")
    st.write("Profile completeness:")
    st.progress(completeness)



def show_trainer_page():
    """Trainer page: integrates Pumpfessor Joe workout builder + calendar."""
    st.header("Trainer")
    st.write("Build your personalized workout and see your training calendar with Pumpfessor Joe 🧠💪")
    st.divider()

    # Use full-width container so embedded modules can expand across the page
    with st.container():
        with st.container(border=True):
            tabs = st.tabs(["Workout builder", "Training calendar"])

            with tabs[0]:
                workout_planner.main()

            with tabs[1]:
                workout_calendar.main()


def show_calorie_tracker_page():
    """Calorie tracker page: integrates ML-based nutrition planner."""
    st.header("Calorie tracker")
    st.divider()

    with st.container():
        with st.container(border=True):
            calorie_tracker.main()

def show_calories_nutrition_page():
    """Calorie tracker page: integrates ML-based nutrition planner."""
    st.header("Calorie tracker")
    st.divider()

    with st.container():
        with st.container(border=True):
            calories_nutrition.main()


def show_nutrition_page():
    """Nutrition adviser page: load logic from nutrition_advisory.py"""
    st.header("Nutrition adviser")
    st.divider()

    with st.container():
        with st.container(border=True):
            # ruft das externe Modul auf
            nutrition_advisory.main()

def show_progress_page():
    """Simple placeholder progress page with a demo chart."""
    st.header("Progress")
    st.divider()

    with st.container():
        with st.container(border=True):
            st.subheader("Demo progress (to be replaced with real data)")

            st.write(
                "This simple chart is a placeholder. "
                "Later, your team can replace it with real workout or calorie data."
            )

            data = {
                "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
                "Workouts": [2, 3, 4, 3],
            }
            df = pd.DataFrame(data).set_index("Week")

            st.bar_chart(df)

            st.info("Your teammates can plug real data into this chart later.")


# =========================================================
# MAIN APP
# =========================================================

def main():
    """Entry point: handle login state and page routing."""
    create_tables()  # make sure DB tables exist

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "login_mode" not in st.session_state:
        st.session_state.login_mode = "login"
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Profile"

    # if not logged in, show auth pages only (with background image)
    if not st.session_state.logged_in:
        # background image on login / register / reset page
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/jpg;base64,{BACKGROUND_IMAGE}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            .block-container {{
                background-color: rgba(255, 255, 255, 0.75);
                border-radius: 1rem;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title("UniFit Coach")
        st.caption("Train smarter. Eat better. Stay consistent. 🌿")
        st.divider()

        mode = st.session_state.login_mode
        if mode == "login":
            show_login_page()
        elif mode == "register":
            show_register_page()
        elif mode == "reset":
            show_reset_password_page()
        return

    # when logged in: remove background image -> plain white
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: none !important;
            background-color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------- SIDEBAR (logo + navigation) ---------------
    if LOGO_IMAGE:
        st.sidebar.markdown(
            f"""
            <div style="text-align: left; padding-top: 1rem; padding-bottom: 1rem;">
                <img src="data:image/png;base64,{LOGO_IMAGE}"
                     style="width: 170px; margin-bottom: 0.5rem;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown("### UniFit Coach")

    st.sidebar.caption("Menu")

    # show logged-in email in the sidebar
    if "user_email" in st.session_state and st.session_state.user_email:
        st.sidebar.caption(f"Logged in as: {st.session_state.user_email}")
        st.sidebar.write("---")

    if st.sidebar.button("👤  Profile"):
        st.session_state.current_page = "Profile"
    if st.sidebar.button("🏋️‍♂️  Trainer"):
        st.session_state.current_page = "Trainer"
    if st.sidebar.button("🔥  Calorie tracker"):
        st.session_state.current_page = "Calorie tracker"
    if st.sidebar.button("🥘  Calories & Nutrition"):
        st.session_state.current_page = "Calories & Nutrition"
    if st.sidebar.button("🥗  Nutrition adviser"):
        st.session_state.current_page = "Nutrition adviser"
    if st.sidebar.button("📈  Progress"):
        st.session_state.current_page = "Progress"

    st.sidebar.write("---")
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.login_mode = "login"
        st.rerun()

    # --------------- MAIN LAYOUT (same structure for all pages) ---------------
    st.title("UniFit Coach")
    st.caption("Train smarter. Eat better. Stay consistent. 🌿")
    if "user_email" in st.session_state and st.session_state.user_email:
        st.write(f"Welcome back, **{st.session_state.user_email}** 👋")
    st.divider()

    page = st.session_state.current_page

    # Pumpfessor Joe helper für die aktuelle Seite
    show_pumpfessor_joe(page)

    # Seiten anzeigen
    if page == "Profile":
        show_profile_page()
    elif page == "Trainer":
        show_trainer_page()
    elif page == "Calorie tracker":
        show_calorie_tracker_page()
    elif page == "Nutrition adviser":
        show_nutrition_page()
    elif page == "Progress":
        show_progress_page()
    # <<< Hier die neue Seite einfügen >>>
    elif page == "Calories & Nutrition":
        show_calories_nutrition_page()



# ---- FINAL CSS OVERRIDES (ensure buttons + sidebar look correct) ----
st.markdown(
    f"""
    <style>
    /* All action buttons (Login, Save profile, etc.) -> white text */
    div.stButton > button,
    div.stButton > button * {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* Sidebar buttons: white by default, green text + border */
    section[data-testid="stSidebar"] div.stButton > button {{
        width: 100% !important;
        background-color: #ffffff !important;
        color: {PRIMARY_GREEN} !important;
        border: 1px solid {PRIMARY_GREEN} !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button * {{
        color: {PRIMARY_GREEN} !important;
    }}

    /* Sidebar buttons on hover/press: green background, white text */
    section[data-testid="stSidebar"] div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button:active,
    section[data-testid="stSidebar"] div.stButton > button:focus {{
        background-color: {PRIMARY_GREEN} !important;
        color: #ffffff !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button:hover *,
    section[data-testid="stSidebar"] div.stButton > button:active *,
    section[data-testid="stSidebar"] div.stButton > button:focus * {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


if __name__ == "__main__":
    main()
