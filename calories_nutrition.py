# calories_nutrition.py
#
# Vollständige Fusion aus:
# - calorie_tracker.py
# - nutrition_advisory.py
# Mit automatischer Einbindung von: database.get_profile + workout_planner.current_workout
#
# Diese Datei wird von app.py über show_calories_nutrition_page() aufgerufen.


import json
import ast
from fractions import Fraction
from collections import Counter
from typing import Optional, List
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Profil-Funktion aus deiner DB ---
from database import get_profile

# --- Rezeptdaten laden wie im Original ---
DATA_URL = (
    "https://huggingface.co/datasets/datahiveai/recipes-with-nutrition"
    "/resolve/main/recipes-with-nutrition.csv"
)

PRIMARY_COLOR = "#007A3D"


# =====================================================================
# 1) MACHINE-LEARNING TEIL (aus calorie_tracker.py)
# =====================================================================

CSV_URL = "https://raw.githubusercontent.com/philippdmt/Protein_and_Calories/refs/heads/main/calories.csv"


def determine_training_type(heart_rate, age):
    if heart_rate >= 0.6 * (220 - age):
        return "Cardio"
    return "Kraft"


@st.cache_data
def load_and_train_model():
    """Kalorienmodell laden + trainieren."""
    try:
        calories = pd.read_csv("calories.csv")
    except FileNotFoundError:
        calories = pd.read_csv(CSV_URL)

    calories["Training_Type"] = calories.apply(
        lambda r: determine_training_type(r["Heart_Rate"], r["Age"]), axis=1
    )

    y = calories["Calories"]

    features = calories.drop(columns=["User_ID", "Heart_Rate", "Body_Temp", "Calories"])
    X = pd.get_dummies(features, columns=["Gender", "Training_Type"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns.tolist()


def grundumsatz(age, weight, height, gender):
    """Mifflin-St Jeor Formel."""
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


# =====================================================================
# 2) REZEPT-MODELL + PARSING (aus nutrition_advisory.py)
# =====================================================================

# Bruchzeichen
UNICODE_FRACTIONS = {
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅐": "1/7", "⅑": "1/9", "⅒": "1/10",
    "⅓": "1/3", "⅔": "2/3",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}


def float_to_fraction_str(x: float, max_denominator: int = 16) -> str:
    f = Fraction(x).limit_denominator(max_denominator)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def parse_quantity_token(t: str):
    t = t.strip()
    if t in UNICODE_FRACTIONS:
        t = UNICODE_FRACTIONS[t]
    try:
        return float(Fraction(t))
    except Exception:
        return None


def split_quantity_from_line(line: str):
    if not line:
        return None, line
    tokens = line.split()
    qty_parts = []
    rest = []
    for i, tok in enumerate(tokens):
        val = parse_quantity_token(tok)
        if val is not None and not rest:
            qty_parts.append(tok)
        else:
            rest = tokens[i:]
            break
    if not qty_parts:
        return None, line
    qty = sum(parse_quantity_token(t) for t in qty_parts)
    return qty, " ".join(rest)


def scale_ingredient_lines(lines, factor: float):
    out = []
    for line in lines:
        qty, rest = split_quantity_from_line(line)
        if qty is None:
            out.append(line)
            continue
        new_qty = qty * factor
        out.append(f"{float_to_fraction_str(new_qty)} {rest}")
    return out


class UserPreferenceModel:
    def __init__(self):
        self.liked = Counter()
        self.disliked = Counter()

    def update_with_rating(self, row: pd.Series, rating: int):
        ings = row.get("ingredients_list", [])
        if rating > 0:
            self.liked.update(ings)
        else:
            self.disliked.update(ings)

    def score(self, row: pd.Series):
        ings = row.get("ingredients_list", [])
        total = 0
        for ing in ings:
            total += self.liked.get(ing, 0)
            total -= self.disliked.get(ing, 0)
        return total


def get_nutrient(nutrient_json, key):
    try:
        d = json.loads(nutrient_json)
        if key in d and "quantity" in d[key]:
            return float(d[key]["quantity"])
    except Exception:
        return None
    return None


def parse_ingredients_for_allergy(x):
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(str(x))
        if isinstance(v, list):
            # {"text": "..."} oder string
            out = []
            for el in v:
                if isinstance(el, dict) and "text" in el:
                    out.append(el["text"].lower())
                else:
                    out.append(str(el).lower())
            return out
    except Exception:
        pass
    return [p.strip().lower() for p in str(x).split(",") if p.strip()]


def parse_ingredient_lines_for_display(x):
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(str(x))
        if isinstance(v, list):
            return [str(el).strip() for el in v]
    except Exception:
        pass
    return [p.strip() for p in str(x).split(",") if p.strip()]


@st.cache_data(show_spinner=True)
def load_and_prepare_recipes():
    df = pd.read_csv(DATA_URL)

    df["protein_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "PROCNT"))
    df["fat_g_total"]     = df["total_nutrients"].apply(lambda x: get_nutrient(x, "FAT"))
    df["carbs_g_total"]   = df["total_nutrients"].apply(lambda x: get_nutrient(x, "CHOCDF"))

    df = df.dropna(subset=["protein_g_total", "fat_g_total", "carbs_g_total", "calories", "servings"])

    df["protein_g"] = df["protein_g_total"] / df["servings"]
    df["fat_g"]     = df["fat_g_total"]     / df["servings"]
    df["carbs_g"]   = df["carbs_g_total"]   / df["servings"]
    df["calories_per_serving"] = df["calories"] / df["servings"]

    df["ingredients_list"] = df["ingredients"].apply(parse_ingredients_for_allergy)
    df["ingredient_lines_parsed"] = df["ingredient_lines"].apply(parse_ingredient_lines_for_display)
    df["ingredient_lines_per_serving"] = df.apply(
        lambda r: scale_ingredient_lines(r["ingredient_lines_parsed"], 1/float(r["servings"])),
        axis=1
    )
    return df


# =====================================================================
# 3) DONUT-CHART (aus calorie_tracker)
# =====================================================================

def donut_chart(consumed, total, title, unit):
    total = max(total, 1)
    consumed = max(consumed, 0)
    remaining = max(total - consumed, 0)

    color = PRIMARY_COLOR if consumed <= total else "#FF0000"

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [consumed, remaining],
        startangle=90,
        counterclock=False,
        colors=[color, "#E0E0E0"],
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )
    ax.set(aspect="equal")
    ax.set_title(title)
    ax.text(0, 0, f"{int(consumed)} / {int(total)} {unit}", ha="center", va="center")
    st.pyplot(fig)


# =====================================================================
# 4) FILTER & PICK RECIPES
# =====================================================================

def filter_recipes(df, diet_pref: str, allergies: List[str]):
    diet_pref = (diet_pref or "No preference").lower()

    base = df.copy()

    if diet_pref == "vegan":
        base = base[base["diet_labels"].astype(str).str.contains("vegan", case=False, na=False)]
    elif diet_pref == "vegetarian":
        base = base[
            base["diet_labels"].astype(str).str.contains("vegetarian", case=False, na=False)
        ]

    if allergies:
        allergies = [a.lower().strip() for a in allergies]
        base = base[base["ingredients_list"].apply(lambda ing:
            not any(any(allg in x for x in ing) for allg in allergies)
        )]

    return base


def pick_meal(df, meal_type: str, calories_target: float, pref_model: Optional[UserPreferenceModel]):
    subset = df[df["meal_type"].astype(str).str.contains(meal_type, case=False, na=False)]
    if subset.empty:
        return None
    subset = subset.copy()

    subset["distance"] = (subset["calories_per_serving"] - calories_target).abs()
    if pref_model:
        subset["score"] = subset.apply(pref_model.score, axis=1)
        subset = subset.sort_values(["score", "distance"], ascending=[False, True])
    else:
        subset = subset.sort_values("distance")

    return subset.head(20).sample(1).iloc[0]


# =====================================================================
# 5) HAUPT-ENTRY FÜR DIE FUSIONIERTE SEITE
# =====================================================================

def main():

    # ------------------------------------------------------
    # A) Profil laden
    # ------------------------------------------------------
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in first.")
        return

    user_id = st.session_state.user_id
    profile = get_profile(user_id)

    if not profile:
        st.error("Could not load your profile.")
        return

    age = profile["age"]
    weight = profile["weight"]
    height = profile["height"]
    gender = profile.get("gender", "Male")
    goal = profile.get("goal", "Maintain")
    allergies = (profile.get("allergies") or "").split(",")
    diet_pref = profile.get("diet_preferences", "No preference")

    # ------------------------------------------------------
    # B) ML-Modell laden
    # ------------------------------------------------------
    try:
        model, feature_cols = load_and_train_model()
    except Exception as e:
        st.error("Model load error.")
        st.exception(e)
        return

    # ------------------------------------------------------
    # C) Rezeptdaten laden (einmal pro App)
    # ------------------------------------------------------
    if "recipes_df" not in st.session_state:
        with st.spinner("Loading recipes..."):
            st.session_state.recipes_df = load_and_prepare_recipes()
    recipes = st.session_state.recipes_df

    # ------------------------------------------------------
    # D) Preference Model initialisieren
    # ------------------------------------------------------
    if "pref_model" not in st.session_state:
        st.session_state.pref_model = UserPreferenceModel()

    # ------------------------------------------------------
    # E) Meal Log initialisieren
    # ------------------------------------------------------
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []

    # ------------------------------------------------------
    # F) Training übernehmen (oder Grundumsatz-only)
    # ------------------------------------------------------
    workout = st.session_state.get("current_workout")

    if workout:
        workout_minutes = workout["minutes"]

        # heuristik: Musclegroup-Workouts = Kraft
        title_lower = workout["title"].lower()
        if any(key in title_lower for key in ["push", "pull", "leg", "upper", "lower", "full body"]):
            training_type = "Kraft"
        else:
            training_type = "Cardio"
    else:
        workout_minutes = 0
        training_type = "Kraft"   # irrelevant, da training_kcal = 0

    # ------------------------------------------------------
    # G) Tagesziel berechnen
    # ------------------------------------------------------
    bmr = grundumsatz(age, weight, height, gender)

    # ML-Feature vorbereiten
    person = {
        "Age": age,
        "Duration": workout_minutes,
        "Weight": weight,
        "Height": height,
        "Gender_Female": 1 if gender.lower() == "female" else 0,
        "Gender_Male":   1 if gender.lower() == "male" else 0,
        "Training_Type_Cardio": 1 if training_type == "Cardio" else 0,
        "Training_Type_Kraft":  1 if training_type == "Kraft" else 0,
    }
    person_df = pd.DataFrame([person]).reindex(columns=feature_cols, fill_value=0)

    if workout:
        training_kcal = float(model.predict(person_df)[0])
    else:
        training_kcal = 0.0

    # Ziel
    if goal.lower() == "bulk":
        target_calories = bmr + training_kcal + 300
        protein_per_kg = 2.0
    elif goal.lower() == "cut":
        target_calories = bmr + training_kcal - 300
        protein_per_kg = 2.2
    else:
        target_calories = bmr + training_kcal
        protein_per_kg = 1.6

    target_calories = max(1200, target_calories)
    target_protein = protein_per_kg * weight

    # ------------------------------------------------------
    # H) UI LAYOUT
    # ------------------------------------------------------
    st.subheader("Calories & Nutrition — Pumpfessor Joe")

    tabs = st.tabs([
        "Daily Calories",
        "Meal Logging",
        "Daily Recipe Plan",
        "Search Recipes",
        "Favourites",
        "Meals Eaten",
    ])

    # ------------------------------------------------------
    # TAB 1: Daily calories
    # ------------------------------------------------------
    with tabs[0]:
        st.write(f"**Workout today:** {workout['title'] if workout else 'No workout'}")
        st.write(f"**Workout duration:** {workout_minutes} min")
        st.write(f"**Training calories:** {round(training_kcal)} kcal")
        st.write(f"**BMR:** {round(bmr)} kcal")

        c1, c2 = st.columns(2)
        with c1:
            donut_chart(
                sum(x["calories"] for x in st.session_state.meal_log),
                target_calories,
                "Calories",
                "kcal"
            )
        with c2:
            donut_chart(
                sum(x["protein"] for x in st.session_state.meal_log),
                target_protein,
                "Protein",
                "g"
            )

    # ------------------------------------------------------
    # TAB 2: Meal Logging
    # ------------------------------------------------------
    with tabs[1]:
        st.subheader("Log a meal")
        with st.form("meal_form"):
            col1, col2, col3 = st.columns([2,1,1])
            name = col1.text_input("Meal name", "Chicken & Rice")
            cal = col2.number_input("Calories", 0, 3000, 500)
            prot = col3.number_input("Protein (g)", 0, 300, 30)
            submit = st.form_submit_button("Add Meal")

        if submit:
            st.session_state.meal_log.append({
                "date_str": date.today().strftime("%d/%m/%Y"),
                "meal": name,
                "calories": float(cal),
                "protein": float(prot),
            })

        if st.button("Reset meals"):
            st.session_state.meal_log = []

        st.write("### Your meals today")
        if st.session_state.meal_log:
            st.table(pd.DataFrame(st.session_state.meal_log))
        else:
            st.info("No meals logged yet.")

    # ------------------------------------------------------
    # TAB 3: Daily recipe plan
    # ------------------------------------------------------
    with tabs[2]:
        st.subheader("Suggested meals for today")

        if st.button("Generate daily plan"):
            user_df = filter_recipes(recipes, diet_pref, allergies)
            breakfast_cal = target_calories * 0.25
            lunch_cal     = target_calories * 0.40
            dinner_cal    = target_calories * 0.35

            plan = {
                "Breakfast": pick_meal(user_df, "breakfast", breakfast_cal, st.session_state.pref_model),
                "Lunch":     pick_meal(user_df, "lunch",     lunch_cal,     st.session_state.pref_model),
                "Dinner":    pick_meal(user_df, "dinner",    dinner_cal,    st.session_state.pref_model),
            }
            st.session_state.daily_plan = plan

        plan = st.session_state.get("daily_plan", None)
        if not plan:
            st.info("Click 'Generate daily plan' to get suggestions.")
        else:
            for meal_name, row in plan.items():
                st.markdown(f"### {meal_name}")
                if row is None:
                    st.warning("No suitable recipe found.")
                else:
                    _show_recipe_card(row, meal_name, recipes)

    # ------------------------------------------------------
    # TAB 4: Search recipes
    # ------------------------------------------------------
    with tabs[3]:
        st.subheader("Search recipes")

        col1,col2,col3 = st.columns(3)
        with col1:
            include = st.text_input("Must include (comma separated)")
        with col2:
            exclude = st.text_input("Exclude (comma separated)")
        with col3:
            meal_type = st.selectbox("Meal type", ["all", "breakfast", "lunch", "dinner"])

        include_list = [x.strip().lower() for x in include.split(",") if x.strip()]
        exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]

        max_cal = st.number_input("Max calories per serving", 0, 3000, 800)

        if st.button("Search"):
            df = filter_recipes(recipes, diet_pref, allergies)
            if meal_type != "all":
                df = df[df["meal_type"].astype(str).str.contains(meal_type, case=False, na=False)]
            if include_list:
                df = df[df["ingredients_list"].apply(
                    lambda ing: all(any(i in x for x in ing) for i in include_list)
                )]
            if exclude_list:
                df = df[df["ingredients_list"].apply(
                    lambda ing: not any(any(e in x for x in ing) for e in exclude_list)
                )]
            df = df[df["calories_per_serving"] <= max_cal]

            if df.empty:
                st.warning("No recipes found.")
            else:
                st.success(f"{len(df)} recipes found (showing first 20).")
                for idx, row in df.head(20).iterrows():
                    _show_recipe_card(row, f"search_{idx}", recipes)

    # ------------------------------------------------------
    # TAB 5: Favourites
    # ------------------------------------------------------
    with tabs[4]:
        st.subheader("Favourite recipes")

        favs = st.session_state.get("favourite_recipes", set())
        if not favs:
            st.info("No favourites yet.")
        else:
            for idx in favs:
                if idx in recipes.index:
                    row = recipes.loc[idx]
                    _show_recipe_card(row, f"fav_{idx}", recipes, mode="favourite")

    # ------------------------------------------------------
    # TAB 6: Meals eaten
    # ------------------------------------------------------
    with tabs[5]:
        st.subheader("Meals eaten")

        if not st.session_state.meal_log:
            st.info("No meals recorded yet.")
        else:
            df = pd.DataFrame(st.session_state.meal_log)
            st.dataframe(df, use_container_width=True)



# =====================================================================
# HELFER FÜR REZEPT-KARTEN
# =====================================================================

def _show_recipe_card(row: pd.Series, meal_name: str, df: pd.DataFrame, mode="default"):
    """Abgespeckte Version aus nutrition_advisory, kompatibel für Fusion."""
    if row is None:
        st.write("No recipe found.")
        return

    name = row["recipe_name"]
    img = row.get("image_url", "")
    calories = row["calories_per_serving"]
    protein = row["protein_g"]
    carbs = row["carbs_g"]
    fat = row["fat_g"]
    ingredients = row.get("ingredient_lines_per_serving", [])

    eaten = name in {x.get("meal") for x in st.session_state.meal_log}

    with st.container():
        col1, col2 = st.columns([1, 5])

        with col1:
            if isinstance(img, str) and img.strip():
                st.image(img, width=200)
            else:
                st.write("No image")

        with col2:
            st.subheader(name)

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Calories", f"{calories:.0f}")
            n2.metric("Protein", f"{protein:.1f}g")
            n3.metric("Carbs", f"{carbs:.1f}g")
            n4.metric("Fat", f"{fat:.1f}g")

            st.markdown("**Ingredients (per serving):**")
            for line in ingredients:
                st.markdown(f"- {line}")

            st.markdown("---")

            # Buttons
            if mode == "favourite":
                if st.button("Remove from favourites", key=f"rmfav_{name}"):
                    st.session_state.favourite_recipes.discard(row.name)
                return

            # Already eaten?
            if not eaten:
                if st.button(f"I ate this ({meal_name})", key=f"eat_{name}"):
                    st.session_state.meal_log.append({
                        "date_str": date.today().strftime("%d/%m/%Y"),
                        "meal": name,
                        "calories": calories,
                        "protein": protein,
                    })
                    st.success("Meal added!")
                if st.button("I don't like this", key=f"skip_{name}"):
                    st.session_state.pref_model.update_with_rating(row, -1)
                return

            # Rating
            if st.button("I liked it", key=f"like_{name}"):
                st.session_state.pref_model.update_with_rating(row, +1)
            if st.button("Save to favourites", key=f"fav_{name}"):
                if "favourite_recipes" not in st.session_state:
                    st.session_state.favourite_recipes = set()
                st.session_state.favourite_recipes.add(row.name)
