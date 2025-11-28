import ast
import json
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

from database import get_profile

PRIMARY_COLOR = "#007A3D"

# --------------------------------------------------------------------
# ML MODEL (from calorie_tracker)
# --------------------------------------------------------------------
CSV_URL = "https://raw.githubusercontent.com/philippdmt/Protein_and_Calories/refs/heads/main/calories.csv"

def determine_training_type(heart_rate, age):
    if heart_rate >= 0.6 * (220 - age):
        return "Cardio"
    return "Kraft"

@st.cache_data
def load_and_train_model():
    """Loads calorie dataset and trains linear regression."""
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
    """Mifflin-St Jeor Formula."""
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


# --------------------------------------------------------------------
# DONUT CHART
# --------------------------------------------------------------------
def donut_chart(consumed, total, title, unit):
    if total <= 0:
        total = 1

    consumed = max(0, consumed)
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


# --------------------------------------------------------------------
# NUTRITION-ADVISORY INGREDIENT PARSING
# --------------------------------------------------------------------
UNICODE_FRACTIONS = {
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅐": "1/7", "⅑": "1/9", "⅒": "1/10",
    "⅓": "1/3", "⅔": "2/3",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}

def float_to_fraction_str(x: float, max_denominator: int = 16):
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
    qty_tokens, rest_tokens = [], []
    for i, tok in enumerate(tokens):
        v = parse_quantity_token(tok)
        if v is not None and not rest_tokens:
            qty_tokens.append(tok)
        else:
            rest_tokens = tokens[i:]
            break
    if not qty_tokens:
        return None, line
    qty = sum(parse_quantity_token(q) for q in qty_tokens)
    rest = " ".join(rest_tokens)
    return qty, rest

def scale_ingredient_lines(lines, factor: float):
    out = []
    for line in lines:
        qty, rest = split_quantity_from_line(line)
        if qty is None:
            out.append(line)
            continue
        new_qty = qty * factor
        out.append(f"{float_to_fraction_str(new_qty)} {rest}".strip())
    return out


# --------------------------------------------------------------------
# LOAD RECIPE DATASET
# --------------------------------------------------------------------
DATA_URL = (
    "https://huggingface.co/datasets/datahiveai/recipes-with-nutrition/"
    "resolve/main/recipes-with-nutrition.csv"
)

@st.cache_data(show_spinner=True)
def load_recipes():
    df = pd.read_csv(DATA_URL)

    def get_nutrient(nutr_json, key):
        try:
            d = json.loads(nutr_json)
            if key in d and "quantity" in d[key]:
                return float(d[key]["quantity"])
        except Exception:
            return None
        return None

    df["protein_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "PROCNT"))
    df["fat_g_total"]     = df["total_nutrients"].apply(lambda x: get_nutrient(x, "FAT"))
    df["carbs_g_total"]   = df["total_nutrients"].apply(lambda x: get_nutrient(x, "CHOCDF"))

    df = df.dropna(subset=["protein_g_total", "fat_g_total", "carbs_g_total", "calories", "servings"])

    df["protein_g"] = df["protein_g_total"] / df["servings"]
    df["fat_g"]     = df["fat_g_total"]     / df["servings"]
    df["carbs_g"]   = df["carbs_g_total"]   / df["servings"]
    df["calories_per_serving"] = df["calories"] / df["servings"]

    # Ingredient parsing
    def parse_ing(x):
        if pd.isna(x):
            return []
        try:
            v = ast.literal_eval(str(x))
            if isinstance(v, list):
                out = []
                for el in v:
                    if isinstance(el, dict) and "text" in el:
                        out.append(el["text"].lower())
                    else:
                        out.append(str(el).lower())
                return out
        except:
            pass
        return [p.strip().lower() for p in str(x).split(",")]

    df["ingredients_list"] = df["ingredients"].apply(parse_ing)

    def parse_lines(x):
        if pd.isna(x):
            return []
        try:
            v = ast.literal_eval(str(x))
            if isinstance(v, list):
                return [str(el).strip() for el in v]
        except:
            pass
        return [p.strip() for p in str(x).split(",")]

    df["ingredient_lines_parsed"] = df["ingredient_lines"].apply(parse_lines)
    df["ingredient_lines_per_serving"] = df.apply(
        lambda r: scale_ingredient_lines(r["ingredient_lines_parsed"], 1/float(r["servings"])),
        axis=1
    )

    return df


# --------------------------------------------------------------------
# FILTERING + PICKING
# --------------------------------------------------------------------
def filter_recipes(df, diet_pref, allergies):
    diet_pref = (diet_pref or "No preference").lower()
    allergies = [a.strip().lower() for a in allergies if a]

    base = df.copy()

    if diet_pref == "vegan":
        base = base[base["diet_labels"].astype(str).str.contains("vegan", case=False, na=False)]
    elif diet_pref == "vegetarian":
        base = base[base["diet_labels"].astype(str).str.contains("vegetarian", case=False, na=False)]

    if allergies:
        base = base[base["ingredients_list"].apply(
            lambda ing: not any(any(a in x for x in ing) for a in allergies)
        )]

    return base

def pick_meal(df, meal_type, target, pref_model):
    subset = df[df["meal_type"].astype(str).str.contains(meal_type, case=False, na=False)]
    if subset.empty:
        return None

    subset = subset.copy()
    subset["cal_diff"] = (subset["calories_per_serving"] - target).abs()

    if pref_model:
        subset["score"] = subset.apply(pref_model.score, axis=1)
        subset = subset.sort_values(["score", "cal_diff"], ascending=[False, True])
    else:
        subset = subset.sort_values("cal_diff")

    return subset.head(20).sample(1).iloc[0]


# --------------------------------------------------------------------
# USER PREFERENCE MODEL
# --------------------------------------------------------------------
class UserPreferenceModel:
    def __init__(self):
        self.liked = Counter()
        self.disliked = Counter()

    def update_with_rating(self, row, rating):
        ings = row.get("ingredients_list", [])
        if rating > 0:
            self.liked.update(ings)
        else:
            self.disliked.update(ings)

    def score(self, row):
        total = 0
        for ing in row.get("ingredients_list", []):
            total += self.liked.get(ing, 0)
            total -= self.disliked.get(ing, 0)
        return total


# --------------------------------------------------------------------
# RECIPE CARD
# --------------------------------------------------------------------
def show_recipe_card(row, key_prefix, meal_name, pref_model, df):
    if row is None:
        st.write("No recipe found.")
        return

    eaten = any(x["meal"] == row["recipe_name"] for x in st.session_state.meal_log)

    with st.container():
        col_img, col_info = st.columns([1, 3])

        with col_img:
            img = row.get("image_url", "")
            if isinstance(img, str) and img.strip():
                st.image(img, width=220)
            else:
                st.write("No image")

        with col_info:
            st.subheader(row["recipe_name"])

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Calories", f"{row['calories_per_serving']:.0f}")
            n2.metric("Protein", f"{row['protein_g']:.1f}g")
            n3.metric("Carbs", f"{row['carbs_g']:.1f}g")
            n4.metric("Fat", f"{row['fat_g']:.1f}g")

            st.markdown("**Ingredients per serving:**")
            for l in row["ingredient_lines_per_serving"]:
                st.markdown(f"- {l}")

            st.markdown("---")

            if not eaten:
                if st.button("I ate this", key=f"eat_{key_prefix}"):
                    st.session_state.meal_log.append({
                        "date_str": date.today().strftime("%d/%m/%Y"),
                        "meal": row["recipe_name"],
                        "calories": row["calories_per_serving"],
                        "protein": row["protein_g"],
                    })
                    st.success("Meal added!")
                if st.button("I don't like this", key=f"skip_{key_prefix}"):
                    pref_model.update_with_rating(row, -1)
                return

            if st.button("I liked it", key=f"like_{key_prefix}"):
                pref_model.update_with_rating(row, +1)
            if st.button("Save to favourites", key=f"fav_{key_prefix}"):
                if "favourite_recipes" not in st.session_state:
                    st.session_state.favourite_recipes = set()
                st.session_state.favourite_recipes.add(row.name)


# --------------------------------------------------------------------
# MAIN APPLICATION PAGE
# --------------------------------------------------------------------
def main():

    # ---------------------------------------------------
    # AUTH
    # ---------------------------------------------------
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in first.")
        return

    user = get_profile(st.session_state.user_id)

    age = user["age"]
    weight = user["weight"]
    height = user["height"]
    gender = user.get("gender", "Male")
    goal = user.get("goal", "Maintain")
    allergies = (user.get("allergies") or "").split(",")
    diet_pref = user.get("diet_preferences", "No preference")

    if age is None or weight is None or height is None:
        st.error("Please complete your profile (age, weight, height).")
        return

    # ---------------------------------------------------
    # SESSION STATE INIT
    # ---------------------------------------------------
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []

    if "pref_model" not in st.session_state:
        st.session_state.pref_model = UserPreferenceModel()

    if "daily_plan" not in st.session_state:
        st.session_state.daily_plan = None

    if "favourite_recipes" not in st.session_state:
        st.session_state.favourite_recipes = set()

    # ---------------------------------------------------
    # LOAD ML MODEL
    # ---------------------------------------------------
    model, feature_cols = load_and_train_model()

    # ---------------------------------------------------
    # WORKOUT IMPORT
    # ---------------------------------------------------
    workout = st.session_state.get("current_workout")
    if workout:
        workout_minutes = workout["minutes"]
        title = workout["title"].lower()
        if any(t in title for t in ["push", "pull", "leg", "upper", "lower", "full"]):
            training_type = "Kraft"
        else:
            training_type = "Cardio"
    else:
        workout_minutes = 0
        training_type = "Kraft"   # irrelevant, kcal = 0

    # ---------------------------------------------------
    # BMR + TARGETS
    # ---------------------------------------------------
    bmr = grundumsatz(age, weight, height, gender)

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
    training_kcal = float(model.predict(person_df)[0]) if workout else 0

    # goal adjustments
    if goal.lower() == "bulk":
        target_calories = bmr + training_kcal + 300
        protein_factor = 2.0
    elif goal.lower() == "cut":
        target_calories = bmr + training_kcal - 300
        protein_factor = 2.2
    else:
        target_calories = bmr + training_kcal
        protein_factor = 1.6

    target_calories = max(target_calories, 1200)
    target_protein = protein_factor * weight

    # ---------------------------------------------------
    # LOAD RECIPE DATASET
    # ---------------------------------------------------
    recipes = load_recipes()

    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------
    tabs = st.tabs([
        "Daily Overview",
        "Daily Recipe Plan",
        "Search Recipes",
        "Favourites"
    ])

    # ---------------------------------------------------
    # TAB 1: DAILY OVERVIEW
    # ---------------------------------------------------
    with tabs[0]:
        st.subheader("Daily Overview")

        col1, col2 = st.columns(2)
        with col1:
            donut_chart(
                sum(m["calories"] for m in st.session_state.meal_log),
                target_calories,
                "Calories",
                "kcal"
            )
        with col2:
            donut_chart(
                sum(m["protein"] for m in st.session_state.meal_log),
                target_protein,
                "Protein",
                "g"
            )

        st.markdown("### Log a meal")
        with st.form("meal_form"):
            c1, c2, c3 = st.columns([2,1,1])
            name = c1.text_input("Meal name", "Chicken & Rice")
            cal = c2.number_input("Calories", 0, 3000, 500)
            prot = c3.number_input("Protein (g)", 0, 200, 30)
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

        st.markdown("### Meals today")
        if st.session_state.meal_log:
            st.table(pd.DataFrame(st.session_state.meal_log))
        else:
            st.info("No meals logged yet.")

    # ---------------------------------------------------
    # TAB 2: DAILY RECIPE PLAN
    # ---------------------------------------------------
    with tabs[1]:
        st.subheader("Daily recipe plan")

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

        plan = st.session_state.get("daily_plan")
        if not plan:
            st.info("Click the button to generate your daily plan.")
        else:
            for meal_name, row in plan.items():
                st.markdown(f"### {meal_name}")
                show_recipe_card(row, meal_name, meal_name, st.session_state.pref_model, recipes)
                st.markdown("---")

    # ---------------------------------------------------
    # TAB 3: SEARCH RECIPES
    # ---------------------------------------------------
    with tabs[2]:
        st.subheader("Search recipes")

        col1, col2, col3 = st.columns(3)
        include = col1.text_input("Must include (comma separated)")
        exclude = col2.text_input("Exclude (comma separated)")
        meal_type = col3.selectbox("Meal type", ["all", "breakfast", "lunch", "dinner"])

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
                for idx, row in df.head(20).iterrows():
                    show_recipe_card(row, f"search{idx}", row["meal_type"], st.session_state.pref_model, recipes)

    # ---------------------------------------------------
    # TAB 4: FAVOURITES
    # ---------------------------------------------------
    with tabs[3]:
        st.subheader("Favourite recipes")

        favs = st.session_state.favourite_recipes
        if not favs:
            st.info("No favourites yet.")
        else:
            for idx in favs.copy():
                if idx not in recipes.index:
                    favs.discard(idx)
                    continue
                row = recipes.loc[idx]
                show_recipe_card(row, f"fav{idx}", row["meal_type"], st.session_state.pref_model, recipes)
