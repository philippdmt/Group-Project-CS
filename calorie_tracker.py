import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from database import get_profile

PRIMARY_COLOR = "#007A3D"
CSV_URL = "https://raw.githubusercontent.com/philippdmt/Protein_and_Calories/refs/heads/main/calories.csv"


def determine_training_type(heart_rate, age):
    if heart_rate >= 0.6 * (220 - age):
        return "Cardio"
    else:
        return "Kraft"


@st.cache_data
def load_and_train_model():
    try:
        calories = pd.read_csv("calories.csv")
    except FileNotFoundError:
        calories = pd.read_csv(CSV_URL)

    calories["Training_Type"] = calories.apply(
        lambda row: determine_training_type(row["Heart_Rate"], row["Age"]), axis=1
    )

    y = calories["Calories"]
    features = calories.drop(columns=["User_ID", "Heart_Rate", "Body_Temp", "Calories"])
    X = pd.get_dummies(features, columns=["Gender", "Training_Type"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns.tolist()


def grundumsatz(age, weight, height, gender):
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161


def donut_chart(consumed, total, title, unit):
    if total <= 0:
        total = 1

    consumed = max(0, consumed)
    remaining = max(total - consumed, 0)
    color = "#007A3D" if consumed <= total else "#FF0000"

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white")
    ax.pie(
        [consumed, remaining],
        startangle=90,
        counterclock=False,
        colors=[color, "#E0E0E0"],
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )
    ax.set(aspect="equal")
    ax.set_title(title)
    ax.text(0, 0, f"{int(consumed)} / {int(total)} {unit}", ha="center", va="center", fontsize=10)
    st.pyplot(fig)


def main():
    st.subheader("Pumpfessor Joe – Nutrition Planner")
    st.write("Automatic calculation of calories and protein based on training and body data.")

    # -------------------------
    # MODEL LOAD
    # -------------------------
    try:
        model, feature_columns = load_and_train_model()
    except Exception as e:
        st.error("Error while loading dataset/model.")
        st.exception(e)
        return

    # -------------------------
    # USER DATA
    # -------------------------
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in first.")
        return

    user_id = st.session_state.user_id
    user = get_profile(user_id)
    if not user:
        st.error("Could not load user profile.")
        return

    age = user["age"]
    weight = user["weight"]
    height = user["height"]
    gender = user.get("gender", "male")
    goal = user.get("goal", "Maintain")

    # -------------------------
    # TRAININGSKALORIEN BERECHNEN
    # -------------------------
    training_kcal = 0
    if "current_workout" in st.session_state:
        current_workout = st.session_state["current_workout"]
        duration = current_workout["minutes"]
        title = current_workout["title"]

        strength_workouts = ["Push Day", "Pull Day", "Leg Day", "Full Body", "Upper Body", "Lower Body"]
        training_type_simple = "Kraft" if title in strength_workouts else "Cardio"

        # Trainingskalorien berechnen
        person = {
            "Age": age,
            "Duration": duration,
            "Weight": weight,
            "Height": height,
            "Gender_Female": 1 if gender.lower() == "female" else 0,
            "Gender_Male": 1 if gender.lower() == "male" else 0,
            "Training_Type_Cardio": 1 if training_type_simple == "Cardio" else 0,
            "Training_Type_Kraft": 1 if training_type_simple == "Kraft" else 0,
        }

        person_df = pd.DataFrame([person])
        person_df = person_df.reindex(columns=feature_columns, fill_value=0)
        training_kcal = float(model.predict(person_df)[0])

    # -------------------------
    # CALCULATIONS
    # -------------------------
    bmr = grundumsatz(age, weight, height, gender)

    if goal.lower() == "bulk":
        target_calories = bmr + training_kcal + 300
        protein_per_kg = 2.0
    elif goal.lower() == "cut":
        target_calories = bmr + training_kcal - 300
        protein_per_kg = 2.2
    else:
        target_calories = bmr + training_kcal
        protein_per_kg = 1.6

    target_calories = max(target_calories, 1200)
    target_protein = protein_per_kg * weight

    # -------------------------
    # MEAL LOGGING
    # -------------------------
    if "meals" not in st.session_state:
        st.session_state.meals = []

    # Berechnung für Charts
    total_cal = sum(m["calories"] for m in st.session_state.meals)
    total_prot = sum(m["protein"] for m in st.session_state.meals)

    # -------------------------
    # DAILY TARGET CHARTS
    # -------------------------
    st.markdown("### Daily targets")
    c1, c2 = st.columns(2)
    with c1:
        donut_chart(total_cal, target_calories, "Calories", "kcal")
    with c2:
        donut_chart(total_prot, target_protein, "Protein", "g")

    # -------------------------
    # LOG MEALS
    # -------------------------
    st.markdown("### Log meals")
    with st.form("meal_form"):
        m1, m2, m3 = st.columns([2, 1, 1])
        meal_name = m1.text_input("Meal name", "Chicken & rice")
        meal_cal = m2.number_input("Calories", 0, 3000, 500)
        meal_prot = m3.number_input("Protein (g)", 0, 200, 30)
        submitted = st.form_submit_button("Add meal")

    if submitted:
        st.session_state.meals.append(
            {"meal": meal_name, "calories": float(meal_cal), "protein": float(meal_prot)}
        )

    if st.button("Reset meals"):
        st.session_state.meals = []

    if st.session_state.meals:
        st.markdown("### Logged meals")
        df_meals = pd.DataFrame(st.session_state.meals)
        # Rundung auf ganze Zahlen
        df_meals["calories"] = df_meals["calories"].round(0).astype(int)
        df_meals["protein"] = df_meals["protein"].round(0).astype(int)
        # Index bei 1 beginnen lassen
        df_meals.index = range(1, len(df_meals) + 1)
        st.table(df_meals)



if __name__ == "__main__":
    main()
