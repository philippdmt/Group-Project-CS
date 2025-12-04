import random            # Python's built-in module for randomness
import pandas as pd      # Pandas for reading / processing the CSV exercise database
import streamlit as st   # Streamlit for the web UI

# Primary brand color used consistently in headings, buttons, etc.
PRIMARY_COLOR = "#007A3D"  # UniFit Coach green

# LOAD DATA
@st.cache_data
def load_exercises(csv_path: str):
    """
    Load and clean the exercise database from a CSV file.
    - Uses a tolerant encoding and separator detection so it works with
      Excel-style CSVs that may use ';' or ',' and include special characters.
    - Normalizes column names.
    - Drops obviously invalid rows (missing exercise or muscle group).
    """
    # Read the CSV:
    # - encoding="latin1" handles many special characters without crashing
    # - sep=None with engine="python" lets pandas auto-detect the separator
    df = pd.read_csv(csv_path, encoding="latin1", sep=None, engine="python")
    # Strip whitespace from column names (e.g. " Exercise " -> "Exercise")
    df.columns = [c.strip() for c in df.columns]
    # Strip whitespace from string values in every object column
    for col in df.columns:
        if df[col].dtype == "object":
            # Convert to string and remove leading/trailing spaces
            df[col] = df[col].astype(str).str.strip()
    # Map different possible column names to a unified schema used in the app
    rename = {}
    for c in df.columns:
        name = c.lower()
        # Any column starting with "exercise" becomes "Exercise"
        if name.startswith("exercise"):
            rename[c] = "Exercise"
        # Any column starting with "equipment" becomes "Equipment Required"
        elif name.startswith("equipment"):
            rename[c] = "Equipment Required"
        # Any column starting with "muscle" becomes "Muscle Group"
        elif name.startswith("muscle"):
            rename[c] = "Muscle Group"
        # Any column starting with "link" becomes "Link"
        elif name.startswith("link"):
            rename[c] = "Link"
    # Actually rename the columns
    df = df.rename(columns=rename)
    # Drop rows where we are missing the exercise name OR muscle group
    df = df.dropna(subset=["Exercise", "Muscle Group"])
    # Some messy CSVs may contain the literal string "nan" instead of real NaN
    df = df[df["Exercise"].str.lower() != "nan"]
    df = df[df["Muscle Group"].str.lower() != "nan"]
    # Return the cleaned dataframe.
    # Because of @st.cache_data, Streamlit will cache this result so the CSV
    # is not re-read on every rerun.
    return df

# MUSCLE INFERENCE (for exercise SELECTION, not soreness UI)
def infer_muscles_from_title(title, all_muscles):
    """
    Infer which muscle groups to focus on based on the workout title
    (e.g. "Push Day", "Pull Day").
    This is used to decide which muscle groups should receive higher scores
    when picking exercises from the database.
    """
    # Work with lowercase to make substring checks easier
    title = title.lower()
    # Mapping from keywords in the workout title -> muscle groups to prioritize
    mapping = {
        # Expanded to include more detailed muscle groups
        "push": ["Chest", "Shoulders", "Traps", "Triceps"],
        "pull": ["Lats", "Upper Back", "Lower Back", "Forearms", "Biceps"],
        "legs": ["Quads", "Hamstrings", "Calves", "Glutes", "Adductor"],
        "upper": [
            "Chest",
            "Shoulders",
            "Traps",
            "Lats",
            "Upper Back",
            "Lower Back",
            "Forearms",
            "Biceps",
            "Triceps",
        ],
        "lower": ["Quads", "Hamstrings", "Calves", "Glutes", "Adductor"],
        "arms": ["Biceps", "Triceps", "Forearms"],
        "chest": ["Chest"],
        "back": ["Lats", "Upper Back", "Lower Back"],
        "shoulder": ["Shoulders"],
        "glute": ["Glutes"],
        "core": ["Abs", "Core"],
        "abs": ["Abs", "Core"],
        "cardio": ["Cardio"],  # in case there is a "Cardio" muscle group in the DB
    }
    # Collect all muscles whose keyword is contained in the title
    found = set()
    for key, muscles in mapping.items():
        if key in title:
            # Only keep muscles that actually exist in the database
            found.update([m for m in muscles if m in all_muscles])
    # If nothing is detected (e.g. custom name), just fall back to first 3 muscles
    if not found:
        return all_muscles[:3]
    return list(found)

# SORENESS OPTIONS PER WORKOUT TYPE (for UI)
def get_soreness_options(title: str) -> list[str]:
    """
    Decide which muscle groups the user can mark as "sore"
    depending on the selected workout type (title).
    This only affects the UI drop-down / multiselect, not the scoring logic.
    """
    # Pre-defined groups used for building soreness options
    push_groups = ["Chest", "Shoulders", "Traps", "Triceps"]
    pull_groups = ["Lats", "Upper Back", "Lower Back", "Forearms", "Biceps"]
    leg_groups = ["Quads", "Hamstrings", "Calves", "Glutes", "Adductor"]
    abs_groups = ["Abs"]
    # For each workout title, specify which groups should be selectable
    mapping = {
        "push day": push_groups,
        "pull day": pull_groups,
        "leg day": leg_groups,
        "lower body": leg_groups,
        "upper body": push_groups + pull_groups,
        "full body": push_groups + pull_groups + leg_groups + abs_groups,
        "cardio": [],  # Cardio: no soreness options
    }
    # Normalize title for lookup
    key = title.lower().strip()
    base = mapping.get(key, [])
    # Remove duplicates while keeping the original order
    seen = set()
    result = []
    for m in base:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result

# "AI" SCORING / WORKOUT LOGIC
def compute_num_exercises(minutes, intensity):
    """
    Roughly estimate how many exercises should be in the workout,
    based on available time and chosen intensity.
    """
    # Base count: about 1 exercise per ~8 minutes, clamped between 3 and 10
    base = max(3, min(10, round(minutes / 8)))
    # Adjust based on intensity: lighter workouts -> fewer exercises,
    # high intensity -> a bit more (but capped).
    if intensity == "Light":
        return max(3, int(base * 0.8))
    if intensity == "Moderate":
        return base
    # "Max effort"
    return min(12, int(base * 1.2))
def score_exercise(row, targets):
    """
    Give each exercise a score based on:
    - whether its primary muscle is in the `targets` list
    - a bit of randomness to avoid always picking the same exercises
    """
    mg = row["Muscle Group"]  # muscle group of this row
    score = 0.0
    # Prefer exercises that hit target muscles
    if mg in targets:
        score += 6.0
    # Add small random noise so the selection varies between runs
    score += random.uniform(-1, 1)
    return score
def sets_reps_rest(intensity):
    """
    Return (sets, reps, rest) recommendations for a given workout intensity.
    """
    if intensity == "Light":
        return 2, "12–15", "45–60 sec"
    if intensity == "Moderate":
        return 3, "8–12", "60–90 sec"
    # "Max effort"
    return 4, "6–10", "90–120 sec"
def apply_soreness_adjustment(exercises, sore_muscles):
    """
    Adjust the workout based on sore muscle groups.
    Rule:
    - For each sore muscle group, remove at most ONE exercise for that group
      from the planned workout.
    """
    # If nothing is sore, return the list unchanged
    if not sore_muscles:
        return exercises
    # Use a set for quick membership checks
    sore_muscles = set(sore_muscles)
    # Track for each sore muscle whether we've already removed an exercise
    removed_for = {m: False for m in sore_muscles}
    final_exercises = []
    for ex in exercises:
        mg = ex["muscle"]  # muscle group of this exercise
        # If this muscle is sore and we haven't removed one yet,
        # skip this exercise (i.e. remove it from the list)
        if mg in removed_for and not removed_for[mg]:
            removed_for[mg] = True
            continue
        # Otherwise keep the exercise
        final_exercises.append(ex)
    return final_exercises
def build_workout_plan(df, title, minutes, sore_muscles, intensity):
    """
    Build a structured workout plan given:
    - df:        cleaned exercise dataframe
    - title:     workout type selected by the user (Push Day, Cardio, etc.)
    - minutes:   available time
    - sore_muscles: list of muscle groups marked as sore in the UI
    - intensity: "Light", "Moderate", or "Max effort"
    Behavior:
    - For Cardio: choose exactly ONE cardio exercise and assign it for the
      full time.
    - For other types: rank all relevant exercises, pick the top N,
      then apply soreness adjustment (remove 1 exercise per sore muscle).
    """
    
    # Special handling for CARDIO WORKOUT
    if title.lower().strip() == "cardio":
        # Filter DB to only cardio exercises: any muscle group containing "cardio"
        df_cardio = df[df["Muscle Group"].str.contains("cardio", case=False, na=False)]
        # If the DB has no cardio entries, return an empty plan
        if df_cardio.empty:
            return []
        # Randomly pick ONE cardio exercise
        row = df_cardio.sample(1).iloc[0]
        # Build single-exercise workout where the "reps" is the total minutes
        exercises = [
            {
                "name": row["Exercise"],
                "muscle": row["Muscle Group"],
                "equipment": row.get("Equipment Required", "–"),
                "sets": 1,
                "reps": f"{minutes} minutes",
                "rest": "As needed",
                "link": row.get("Link", ""),
            }
        ]
        # Cardio ignores soreness, so just return here
        return exercises
    
    # NORMAL (RESISTANCE) WORKOUT FLOW
    # All available muscle groups in the DB
    all_muscles = sorted(df["Muscle Group"].unique())
    # Decide which muscles to focus on based on workout title
    targets = infer_muscles_from_title(title, all_muscles)
    # Safety: if for some reason DB is empty, return no workout
    if df.empty:
        return []
    # Work on a copy so we don't modify the original dataframe
    df = df.copy()
    # Compute a "score" for each exercise row based on targets + randomness
    df["score"] = df.apply(
        lambda r: score_exercise(r, targets), axis=1
    )
    # Highest scores first
    df = df.sort_values("score", ascending=False)
    # Decide how many exercises we want
    num = min(compute_num_exercises(minutes, intensity), len(df))
    # Get recommended sets, reps, and rest based on intensity
    sets, reps, rest = sets_reps_rest(intensity)
    # Build the list of exercise dicts that will be used by the UI
    exercises = []
    for _, r in df.head(num).iterrows():
        exercises.append(
            {
                "name": r["Exercise"],
                "muscle": r["Muscle Group"],
                "equipment": r.get("Equipment Required", "–"),
                "sets": sets,
                "reps": reps,
                "rest": rest,
                "link": r.get("Link", ""),
            }
        )
    # Apply soreness rule AFTER constructing the list:
    # one fewer exercise per sore muscle group
    exercises = apply_soreness_adjustment(exercises, sore_muscles)
    return exercises

# FLASHCARD VIEW
def show_flashcards():
    """
    Display the current exercise as a "flashcard" and
    provide navigation buttons.
    - Uses st.session_state.workout (list of exercises).
    - Uses st.session_state.current_card for the index.
    - Hides the "Previous" button on the first card.
    - Changes "Next" to "Complete workout" on the last card.
    """
    state = st.session_state
    exercises = state.workout          # full workout list
    idx = state.current_card           # index of current exercise
    total = len(exercises)             # total number of exercises
    ex = exercises[idx]                # current exercise dict
    # Header: show progress "Exercise X of Y"
    st.write(f"### Exercise {idx + 1} of {total}")
    # Progress bar: fraction of workout completed
    st.progress((idx + 1) / total)
    # Main flashcard with exercise info in styled HTML
    st.markdown(
        f"""
<div style="padding:25px; border-radius:22px; background:white;
            box-shadow:0 6px 14px rgba(0,0,0,0.12); margin-top:20px;">
  <h2 style="color:{PRIMARY_COLOR}; margin-top:0;">{ex['name']}</h2>
  <ul style="font-size:16px; line-height:1.7;">
    <li><b>Muscle trained:</b> {ex['muscle']}</li>
    <li><b>Equipment needed:</b> {ex['equipment']}</li>
    <li><b>Your goal:</b> {ex['sets']} sets × {ex['reps']} reps</li>
    <li><b>Rest between sets:</b> {ex['rest']}</li>
  </ul>
  {f'<a href="{ex["link"]}" target="_blank" style="color:{PRIMARY_COLOR}; font-weight:bold;">Video exercise demonstration</a>'
      if isinstance(ex["link"], str) and ex["link"].startswith("http") else ""}
</div>
""",
        unsafe_allow_html=True,
    )
    # Flags for first/last exercise to control button behavior
    is_first = idx == 0
    is_last = idx == total - 1
    # Layout for navigation buttons: previous on the left, next/complete on the right
    col_prev, col_next = st.columns(2)
    # Left column: show "Previous Exercise" button only if we are not on the first exercise
    with col_prev:
        if not is_first:
            if st.button("⬅️ Previous Exercise"):
                state.current_card -= 1
                st.rerun()  # rerun app to render new exercise
    # Right column: "Next Exercise" or "Complete workout" on the last card
    with col_next:
        next_label = "Complete workout ✅" if is_last else "Next Exercise 👉"
        if st.button(next_label):
            if is_last:
                # Last exercise: mark workout as finished
                state.finished = True
            else:
                # Otherwise move to the next exercise
                state.current_card += 1
            st.rerun()

# COMPLETION SCREEN
def show_completion():
    """
    Show a summary page once the workout is finished.
    Uses st.session_state.workout to list all performed exercises.
    """
    state = st.session_state
    # Big title and congratulatory message
    st.markdown(
        f"<h2 style='color:{PRIMARY_COLOR};'>Workout Completed! 🎉</h2>",
        unsafe_allow_html=True,
    )
    st.success("Joe the Pumpfessor is proud of you! 💪🔥")
    st.write("### Your full workout summary:")
    # Summary card for each exercise in the workout
    for ex in state.workout:
        st.markdown(
            f"""
<div style="padding:18px; border-radius:16px; border:2px solid {PRIMARY_COLOR};
            background:#F4FFF7; margin:10px 0;">
  <h4 style='color:{PRIMARY_COLOR}; margin-top:0;'>{ex['name']}</h4>
  <p><b>Muscle:</b> {ex['muscle']}</p>
  <p><b>Equipment:</b> {ex['equipment']}</p>
  <p><b>Sets/Reps:</b> {ex['sets']} × {ex['reps']}</p>
  <p><b>Rest:</b> {ex['rest']}</p>
  {f'<a href="{ex["link"]}" target="_blank">Video exercise demonstration</a>'
      if isinstance(ex["link"], str) and ex["link"].startswith("http") else ""}
</div>
""",
            unsafe_allow_html=True,
        )
    # Button to go back to the workout builder and reset workout-related state
    if st.button("Back to workout builder ↩️"):
        for key in ["workout", "current_card", "finished"]:
            if key in state:
                del state[key]
        st.rerun()

# PUBLIC ENTRY POINT USED BY app.py
def main():
    """
    Entry point used by the main Streamlit app (app.py).
    Responsibilities:
    - Load the exercise database.
    - If a workout is in progress, show flashcards / completion.
    - Otherwise render the workout builder form.
    """
    state = st.session_state
    # Load CSV – must live in same folder as app.py
    df = load_exercises("CS Workout Exercises Database CSV.csv")
    # If a workout is already generated and not finished, show flashcards
    if "workout" in state and not state.get("finished", False):
        show_flashcards()
        return
    # If workout is marked as finished, show completion summary
    if state.get("finished", False):
        show_completion()
        return
    # ----- Workout builder form (initial view) -----
    st.subheader("Build a workout with Pumpfessor Joe")
    st.caption("Answer a few questions and get a suggested workout plan.")
    # Main workout type selector
    workout_options = [
        "Push Day",
        "Pull Day",
        "Leg Day",
        "Full Body",
        "Upper Body",
        "Lower Body",
        "Cardio",
    ]
    title = st.selectbox("Choose your workout type:", workout_options, index=0)
    # Slider for time available
    minutes = st.slider("How many minutes do you have?", 15, 120, 45, 5)
    # Store current workout meta info for other parts of the app (e.g. calorie tracker)
    st.session_state["current_workout"] = {
        "title": title,
        "minutes": minutes,
    }
    st.markdown(
        f"<p style='color:{PRIMARY_COLOR};'><b>Are you sore anywhere?</b></p>",
        unsafe_allow_html=True,
    )
    # CSS override so selected multiselect tags appear in green with white text
    st.markdown(
        f"""
        <style>
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {PRIMARY_COLOR} !important;
            color: white !important;
        }}
        .stMultiSelect [data-baseweb="tag"] span {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Determine soreness options based on selected workout type
    soreness_options = get_soreness_options(title)
    # Multiselect only if there are options for this workout type
    if not soreness_options:
        st.caption("No sore muscle selection for this workout type.")
        sore_groups = []
    else:
        sore_groups = st.multiselect(
            "Select sore muscle groups:",
            options=soreness_options,
        )
    # Training intensity
    intensity = st.selectbox("Intensity:", ["Light", "Moderate", "Max effort"], 1)
    # Button to generate the workout
    if st.button("Generate Workout"):
        # Build the workout plan based on the chosen parameters
        workout = build_workout_plan(df, title, minutes, sore_groups, intensity)

        if not workout:
            # If no exercises could be found, show a helpful warning
            st.warning(
                "No suitable exercises found. Try changing time, intensity, or soreness selection."
            )
        else:
            # Save workout & flashcard state, then rerun to show flashcards
            state.workout = workout
            state.current_card = 0
            state.finished = False
            st.rerun()
