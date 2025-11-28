# workout_planner.py  (or rename to trainer_module.py)

import random
import pandas as pd
import streamlit as st

PRIMARY_COLOR = "#007A3D"  # UniFit Coach green


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_exercises(csv_path: str):
    """
    Load and clean exercise database from CSV.

    Uses a tolerant encoding and lets pandas guess the separator so it
    works with Excel-style CSVs that use ; or , and contain special
    characters.
    """
    # Let pandas auto-detect the separator, use a tolerant encoding
    df = pd.read_csv(csv_path, encoding="latin1", sep=None, engine="python")

    # strip column names and values
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # unify column names
    rename = {}
    for c in df.columns:
        name = c.lower()
        if name.startswith("exercise"):
            rename[c] = "Exercise"
        elif name.startswith("equipment"):
            rename[c] = "Equipment Required"
        elif name.startswith("muscle"):
            rename[c] = "Muscle Group"
        elif name.startswith("link"):
            rename[c] = "Link"

    df = df.rename(columns=rename)

    # drop rows without exercise / muscle info
    df = df.dropna(subset=["Exercise", "Muscle Group"])
    df = df[df["Exercise"].str.lower() != "nan"]
    df = df[df["Muscle Group"].str.lower() != "nan"]

    return df


# ---------------------------------------------------
# MUSCLE INFERENCE
# ---------------------------------------------------
def infer_muscles_from_title(title, all_muscles):
    title = title.lower()

    mapping = {
        "push": ["Chest", "Triceps", "Shoulders"],
        "pull": ["Back", "Biceps"],
        "legs": ["Legs", "Quads", "Hamstrings", "Glutes"],
        "upper": ["Chest", "Shoulders", "Back", "Arms"],
        "lower": ["Legs", "Quads", "Hamstrings", "Glutes"],
        "arms": ["Biceps", "Triceps", "Forearms"],
        "chest": ["Chest"],
        "back": ["Back"],
        "shoulder": ["Shoulders"],
        "glute": ["Glutes"],
        "core": ["Abs", "Core"],
        "abs": ["Abs", "Core"],
    }

    found = set()
    for key, muscles in mapping.items():
        if key in title:
            found.update([m for m in muscles if m in all_muscles])

    if not found:
        # fallback: first 3 muscles from list
        return all_muscles[:3]

    return list(found)


# ---------------------------------------------------
# "AI" SCORING / WORKOUT LOGIC
# ---------------------------------------------------
def compute_num_exercises(minutes, intensity):
    base = max(3, min(10, round(minutes / 8)))
    if intensity == "Light":
        return max(3, int(base * 0.8))
    if intensity == "Moderate":
        return base
    return min(12, int(base * 1.2))


def score_exercise(row, targets, soreness, intensity):
    mg = row["Muscle Group"]
    score = 0.0

    # prefer target muscles
    if mg in targets:
        score += 6.0

    # avoid sore muscles
    score -= soreness.get(mg, 0) * 1.6

    # little randomness
    score += random.uniform(-1, 1)

    return score


def sets_reps_rest(intensity):
    if intensity == "Light":
        return 2, "12–15", "45–60 sec"
    if intensity == "Moderate":
        return 3, "8–12", "60–90 sec"
    return 4, "6–10", "90–120 sec"


def build_workout_plan(df, title, minutes, wishes, soreness, intensity):
    all_muscles = sorted(df["Muscle Group"].unique())
    targets = infer_muscles_from_title(title, all_muscles)

    # completely avoid very sore muscles
    very_sore = [m for m, s in soreness.items() if s >= 8]
    df = df[~df["Muscle Group"].isin(very_sore)]

    if df.empty:
        return []

    df = df.copy()
    df["score"] = df.apply(
        lambda r: score_exercise(r, targets, soreness, intensity), axis=1
    )
    df = df.sort_values("score", ascending=False)

    num = min(compute_num_exercises(minutes, intensity), len(df))
    sets, reps, rest = sets_reps_rest(intensity)

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

    return exercises


# ---------------------------------------------------
# FLASHCARD VIEW
# ---------------------------------------------------
def show_flashcards():
    state = st.session_state
    exercises = state.workout
    idx = state.current_card
    total = len(exercises)
    ex = exercises[idx]

    st.write(f"### Exercise {idx + 1} of {total}")
    st.progress((idx + 1) / total)

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

    if st.button("Next Exercise 👉"):
        state.current_card += 1
        if state.current_card >= total:
            state.finished = True
        st.rerun()


# ---------------------------------------------------
# COMPLETION SCREEN
# ---------------------------------------------------
def show_completion():
    state = st.session_state

    st.markdown(
        f"<h2 style='color:{PRIMARY_COLOR};'>Workout Completed! 🎉</h2>",
        unsafe_allow_html=True,
    )
    st.success("Joe the Pumpfessor is proud of you! 💪🔥")

    st.write("### Your full workout summary:")

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

    if st.button("Back to workout builder ↩️"):
        for key in ["workout", "current_card", "finished"]:
            if key in state:
                del state[key]
        st.rerun()


# ---------------------------------------------------
# PUBLIC ENTRY POINT USED BY app.py
# ---------------------------------------------------
def main():
    """
    Render the workout builder inside the Trainer page.

    app.py calls this inside show_trainer_page(), so we:
    - DO NOT call st.set_page_config() here
    - Just draw the trainer UI.
    """
    state = st.session_state

    # Load CSV – must live in same folder as app.py
    df = load_exercises("CS Workout Exercises Database CSV.csv")
    muscles = sorted(df["Muscle Group"].unique())

    # If a workout is already generated, show flashcards or completion
    if "workout" in state and not state.get("finished", False):
        show_flashcards()
        return

    if state.get("finished", False):
        show_completion()
        return

    # Otherwise show the workout builder form
    st.subheader("Build a workout with Pumpfessor Joe")
    st.caption("Answer a few questions and get a suggested workout plan.")

    workout_options = ["Push Day", "Pull Day", "Leg Day", "Full Body", "Upper Body", "Lower Body"]
    title = st.selectbox("Choose your workout type:", workout_options, index=0)
    minutes = st.slider("How many minutes do you have?", 15, 120, 45, 5)

    # Speichern für den Calorie Tracker
    st.session_state["current_workout"] = {
        "title": title,
        "minutes": minutes
}


    st.markdown(
        f"<p style='color:{PRIMARY_COLOR};'><b>Are you sore anywhere?</b></p>",
        unsafe_allow_html=True,
    )

    soreness = {}
    sore_groups = st.multiselect("Select sore muscle groups:", muscles)
    for m in sore_groups:
        soreness[m] = st.slider(f"Soreness in {m}", 1, 10, 5)

    wishes = st.text_area("Additional wishes (optional):")
    intensity = st.selectbox("Intensity:", ["Light", "Moderate", "Max effort"], 1)

    if st.button("Generate Workout"):
        workout = build_workout_plan(df, title, minutes, wishes, soreness, intensity)
        if not workout:
            st.warning("No suitable exercises found. Try changing time, intensity or soreness.")
        else:
            state.workout = workout
            state.current_card = 0
            state.finished = False
            st.rerun()

