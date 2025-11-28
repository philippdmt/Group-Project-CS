import ast
import json
from fractions import Fraction
from collections import Counter
from typing import List, Optional
from datetime import date
from database import get_profile

import pandas as pd
import streamlit as st

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Nutrition Advisory",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: CSS für volle Breite von Markdown/Container
st.markdown(
    """
    <style>
    .css-1d391kg, .css-1avcm0n {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# CONFIG
# =============================================================================
DATA_URL = (
    "https://huggingface.co/datasets/datahiveai/recipes-with-nutrition"
    "/resolve/main/recipes-with-nutrition.csv"
)

HIGH_PROTEIN_MIN = 25
MAX_CALORIES_PER_SERVING = 800

# =============================================================================
# USER PREFERENCE MODEL
# =============================================================================
class UserPreferenceModel:
    def __init__(self):
        self.liked_ingredients = Counter()
        self.disliked_ingredients = Counter()

    def update_with_rating(self, recipe_row: pd.Series, rating: int):
        ings = recipe_row.get("ingredients_list", [])
        if rating > 0:
            self.liked_ingredients.update(ings)
        elif rating < 0:
            self.disliked_ingredients.update(ings)

    def score_recipe(self, recipe_row: pd.Series) -> float:
        ings = recipe_row.get("ingredients_list", [])
        score = 0.0
        for ing in ings:
            score += self.liked_ingredients.get(ing, 0)
            score -= self.disliked_ingredients.get(ing, 0)
        return score

# =============================================================================
# INGREDIENT PARSING
# =============================================================================
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

def parse_quantity_token(token: str):
    token = token.strip()
    if token in UNICODE_FRACTIONS:
        token = UNICODE_FRACTIONS[token]
    try:
        return float(Fraction(token))
    except Exception:
        return None

def split_quantity_from_line(line: str):
    if not line:
        return None, line
    tokens = line.split()
    qty_tokens = []
    rest_tokens = []
    for i, tok in enumerate(tokens):
        val = parse_quantity_token(tok)
        if val is not None and not rest_tokens:
            qty_tokens.append(tok)
        else:
            rest_tokens = tokens[i:]
            break
    if not qty_tokens:
        return None, line

    qty = sum(parse_quantity_token(t) for t in qty_tokens)
    rest = " ".join(rest_tokens).strip()
    return qty, rest

def scale_ingredient_lines(lines, factor: float):
    scaled = []
    for line in lines:
        qty, rest = split_quantity_from_line(line)
        if qty is None:
            scaled.append(line)
            continue
        new_qty = qty * factor
        scaled.append(f"{float_to_fraction_str(new_qty)} {rest}".strip())
    return scaled

# =============================================================================
# DATA HELPERS
# =============================================================================
def get_nutrient(nutrient_json, key):
    try:
        data = json.loads(nutrient_json)
        if key in data and "quantity" in data[key]:
            return float(data[key]["quantity"])
    except Exception:
        return None
    return None

def parse_ingredients_for_allergy(x):
    if pd.isna(x): return []
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [str(v.get("text", v)).lower() for v in val]
    except Exception:
        pass
    return [p.strip().lower() for p in str(x).split(",") if p.strip()]

def parse_ingredient_lines_for_display(x):
    if pd.isna(x): return []
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [str(v).strip() for v in val]
    except Exception:
        pass
    return [p.strip() for p in str(x).split(",") if p.strip()]

@st.cache_data(show_spinner=True)
def load_and_prepare_data(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    df["protein_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "PROCNT"))
    df["fat_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "FAT"))
    df["carbs_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "CHOCDF"))

    df = df.dropna(subset=["protein_g_total", "fat_g_total", "carbs_g_total", "calories", "servings"])

    df["calories"] = df["calories"].astype(float)
    df["servings"] = pd.to_numeric(df["servings"], errors="coerce").fillna(1).clip(lower=1)

    df["calories_per_serving"] = df["calories"] / df["servings"]
    df["protein_g"] = df["protein_g_total"] / df["servings"]
    df["fat_g"] = df["fat_g_total"] / df["servings"]
    df["carbs_g"] = df["carbs_g_total"] / df["servings"]

    fitness_df = df[
        (df["protein_g"] >= HIGH_PROTEIN_MIN) &
        (df["calories_per_serving"] <= MAX_CALORIES_PER_SERVING)
    ].copy()

    fitness_df["ingredients_list"] = fitness_df["ingredients"].apply(parse_ingredients_for_allergy)
    fitness_df["ingredient_lines_parsed"] = fitness_df["ingredient_lines"].apply(parse_ingredient_lines_for_display)
    fitness_df["ingredient_lines_per_serving"] = fitness_df.apply(
        lambda row: scale_ingredient_lines(row["ingredient_lines_parsed"], 1.0 / row["servings"]),
        axis=1
    )
    return fitness_df

# =============================================================================
# FILTER / SEARCH / PLAN
# =============================================================================
def filter_by_preferences(df: pd.DataFrame, diet_pref: str, allergies: List[str]) -> pd.DataFrame:
    diet_pref = diet_pref.lower()
    allergies = [a.lower() for a in allergies]

    res = df.copy()

    if diet_pref == "vegan":
        res = res[res["diet_labels"].str.contains("vegan", case=False, na=False)]
    elif diet_pref == "vegetarian":
        res = res[
            res["diet_labels"].str.contains("vegetarian", case=False, na=False) |
            res["diet_labels"].str.contains("vegan", case=False, na=False)
        ]

    if allergies:
        res = res[res["ingredients_list"].apply(lambda ing: not any(a in x for a in allergies for x in ing))]

    return res

def search_recipes(
    df: pd.DataFrame,
    include_ingredients: List[str],
    exclude_ingredients: List[str],
    meal_type: str,
    max_calories: Optional[float],
    diet_pref: str,
    allergies: List[str],
    pref_model: Optional[UserPreferenceModel],
) -> pd.DataFrame:

    base = filter_by_preferences(df, diet_pref, allergies)
    include = [i.lower() for i in include_ingredients]
    exclude = [i.lower() for i in exclude_ingredients]

    if meal_type != "all":
        base = base[base["meal_type"].str.contains(meal_type, case=False, na=False)]
    if max_calories:
        base = base[base["calories_per_serving"] <= max_calories]
    if include:
        base = base[base["ingredients_list"].apply(lambda ing: all(any(i in x for x in ing) for i in include))]
    if exclude:
        base = base[base["ingredients_list"].apply(lambda ing: not any(any(e in x for x in ing) for e in exclude))]

    if pref_model is not None:
        base = base.copy()
        base["score"] = base.apply(pref_model.score_recipe, axis=1)
        base = base.sort_values("score", ascending=False)

    return base

def pick_meal(df: pd.DataFrame, meal_type: str, target_cal: float, training_goal: str, pref_model: Optional[UserPreferenceModel]):
    base = df[df["meal_type"].str.contains(meal_type, case=False, na=False)]
    if base.empty:
        return None

    base = base.copy()
    base["cal_diff"] = (base["calories_per_serving"] - target_cal).abs()

    if training_goal == "strength":
        base = base.sort_values(["protein_g", "cal_diff"], ascending=[False, True])
    elif training_goal == "endurance":
        base = base.sort_values(["carbs_g", "cal_diff"], ascending=[False, True])
    else:
        base = base.sort_values("cal_diff")

    if pref_model:
        base["score"] = base.apply(pref_model.score_recipe, axis=1)
        base = base.sort_values(["score", "cal_diff"], ascending=[False, True])

    return base.head(20).sample(1).iloc[0]

def recommend_daily_plan(df, daily_calories, training_goal, diet_pref, allergies, pref_model):
    user_df = filter_by_preferences(df, diet_pref, allergies)
    return {
        "Breakfast": (pick_meal(user_df, "breakfast", daily_calories * 0.25, training_goal, pref_model), daily_calories * 0.25),
        "Lunch":     (pick_meal(user_df, "lunch",     daily_calories * 0.40, training_goal, pref_model), daily_calories * 0.40),
        "Dinner":    (pick_meal(user_df, "dinner",    daily_calories * 0.35, training_goal, pref_model), daily_calories * 0.35),
    }

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    if "pref_model" not in st.session_state:
        st.session_state.pref_model = UserPreferenceModel()
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []
    if "daily_plan" not in st.session_state:
        st.session_state.daily_plan = None
    if "eaten_today" not in st.session_state:
        st.session_state.eaten_today = set()
    if "rating_stage" not in st.session_state:
        st.session_state.rating_stage = {}
    if "ct_meals" not in st.session_state:
        st.session_state.ct_meals = []

# =============================================================================
# MEAL LOGGING
# =============================================================================
def log_meal(row: pd.Series, meal_name: str):
    st.session_state.meal_log.append({
        "date_str": date.today().strftime("%d/%m/%Y"),
        "recipe_name": row["recipe_name"],
        "calories": row["calories_per_serving"],
        "protein": row["protein_g"],
    })
    st.session_state.eaten_today.add(row["recipe_name"])


# =============================================================================
# RECIPE CARD RENDERING
# =============================================================================
def show_recipe_card(
    row: pd.Series,
    key_prefix: str,
    meal_name: str,
    mode: str = "default",
    df: Optional[pd.DataFrame] = None,
    profile: Optional[dict] = None,
    pref_model: Optional[UserPreferenceModel] = None,
    meal_target_calories: Optional[float] = None,
):
    if row is None:
        st.write("No suitable recipe found.")
        return

    recipe_name = row["recipe_name"]
    eaten = recipe_name in st.session_state.eaten_today
    rating_stage = st.session_state.rating_stage.get(recipe_name, "none")

    with st.container():
        col_left, col_right = st.columns([1, 5])

        # Image
        with col_left:
            img_url = row.get("image_url", "")
            if img_url:
                st.image(img_url, width=240)
            else:
                st.write("No image available")

        with col_right:
            st.subheader(recipe_name)

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Calories", f"{row['calories_per_serving']:.0f}")
            n2.metric("Protein", f"{row['protein_g']:.1f} g")
            n3.metric("Carbs", f"{row['carbs_g']:.1f} g")
            n4.metric("Fat", f"{row['fat_g']:.1f} g")

            st.markdown("**Ingredients (per serving)**")
            for line in row.get("ingredient_lines_per_serving", []):
                st.markdown(f"- {line}")

            st.markdown("---")

            # Favourite mode
            if mode == "favourite":
                if st.button("Remove from favourite recipes", key=f"remove_{key_prefix}"):
                    st.session_state.favourite_recipes.discard(row.name)
                return

            # If not eaten
            if not eaten:
                b1, b2 = st.columns(2)
                if b1.button("I have eaten this", key=f"eat_{key_prefix}"):
                    log_meal(row, meal_name)
                    st.session_state.rating_stage[recipe_name] = "none"

                if b2.button("I don't like this meal", key=f"skip_{key_prefix}"):
                    if df is not None and meal_target_calories is not None:
                        new_row = pick_meal(df, meal_name, meal_target_calories, "", pref_model)
                        if new_row is not None:
                            st.session_state.daily_plan[meal_name] = (new_row, meal_target_calories)
                return

            # Rating
            if rating_stage == "none":
                b1, b2 = st.columns(2)
                if b1.button("I liked this meal", key=f"like_{key_prefix}"):
                    if pref_model:
                        pref_model.update_with_rating(row, +1)
                    st.session_state.rating_stage[recipe_name] = "liked"

                if b2.button("I didn't like this meal", key=f"dislike_{key_prefix}"):
                    if pref_model:
                        pref_model.update_with_rating(row, -1)
                    st.session_state.rating_stage[recipe_name] = "disliked"

            elif rating_stage == "liked":
                b1, b2 = st.columns(2)
                if b1.button("Save in favourites", key=f"fav_{key_prefix}"):
                    st.session_state.favourite_recipes.add(row.name)
                    st.session_state.rating_stage[recipe_name] = "liked_saved"

                if b2.button("Don't save in favourites", key=f"nofav_{key_prefix}"):
                    st.session_state.rating_stage[recipe_name] = "liked_nosave"


# =============================================================================
# MAIN APP
# =============================================================================
def main(df=None):
    init_session_state()

    # Load DF
    if df is None:
        df = st.session_state.get("recipes_df")
    if df is None:
        st.info("Loading recipe data...")
        df = load_and_prepare_data(DATA_URL)
        st.session_state.recipes_df = df

    # Local placeholder profile (real profile is only needed in calorie tracker)
    profile = {
        "daily_calories": 2000,
        "training_goal": "strength",
        "diet_pref": "omnivore",
        "allergies": [],
    }

    tab_caltracker, tab_suggested, tab_search, tab_fav, tab_log = st.tabs([
        "Calorie Tracker",
        "Suggested recipes",
        "Search recipes",
        "Favourite recipes",
        "Meals eaten",
    ])

    # ---------------------- SUGGESTED RECIPES ----------------------
    with tab_suggested:
        st.subheader("Suggested recipes for today")

        if st.button("Generate daily plan"):
            st.session_state.daily_plan = recommend_daily_plan(
                df,
                profile["daily_calories"],
                profile["training_goal"],
                profile["diet_pref"],
                profile["allergies"],
                st.session_state.pref_model,
            )

        plan = st.session_state.daily_plan
        if plan is None:
            st.info("Click 'Generate daily plan' to get recommendations.")
        else:
            for meal_name, (row, target_cal) in plan.items():
                st.markdown(f"### {meal_name}")
                show_recipe_card(
                    row,
                    f"plan_{meal_name}",
                    meal_name,
                    "default",
                    df,
                    profile,
                    st.session_state.pref_model,
                    target_cal,
                )

    # ---------------------- SEARCH RECIPES ----------------------
    with tab_search:
        st.subheader("Search recipes")

        col1, col2, col3 = st.columns(3)
        include_text = col1.text_input("Must include ingredients (comma separated)")
        exclude_text = col2.text_input("Exclude ingredients (comma separated)")
        meal_type = col3.selectbox("Meal type", ["all", "breakfast", "lunch", "dinner"])

        max_cal = st.number_input("Max calories per serving", 0, 3000, 800)

        if st.button("Search"):
            include_ingredients = [x.strip() for x in include_text.split(",") if x.strip()]
            exclude_ingredients = [x.strip() for x in exclude_text.split(",") if x.strip()]

            st.session_state.search_results = search_recipes(
                df,
                include_ingredients,
                exclude_ingredients,
                meal_type,
                max_cal,
                profile["diet_pref"],
                profile["allergies"],
                st.session_state.pref_model,
            )

        results = st.session_state.get("search_results")
        if results is None:
            st.info("Set filters and click 'Search'.")
        elif results.empty:
            st.warning("No recipes matched your filters.")
        else:
            st.write(f"Found {len(results)} recipes (showing first 20):")
            for idx, row in results.head(20).iterrows():
                show_recipe_card(
                    row,
                    f"search_{idx}",
                    "Search",
                    "default",
                    df,
                    None,
                    st.session_state.pref_model,
                )

    # ---------------------- FAVOURITE RECIPES ----------------------
    with tab_fav:
        st.subheader("Favourite recipes")

        fav_indices = list(st.session_state.get("favourite_recipes", []))
        if not fav_indices:
            st.write("You have no favourite recipes yet.")
        else:
            for idx in fav_indices:
                if idx in df.index:
                    row = df.loc[idx]
                    show_recipe_card(row, f"fav_{idx}", "Favourite", "favourite")

    # ---------------------- MEALS EATEN ----------------------
    with tab_log:
        st.subheader("Meals eaten today")

        meal_log = st.session_state.meal_log
        if not meal_log:
            st.info("You have not logged any meals yet.")
        else:
            df_log_simple = pd.DataFrame(meal_log)[["date_str", "recipe_name", "calories", "protein"]]
            df_log_simple.columns = ["Date", "Meal", "Calories", "Protein (g)"]

            st.table(df_log_simple)

    # =============================================================================
    # CALORIE TRACKER (Training automatisch aus Workout Planner)
    # =============================================================================
    with tab_caltracker:
        st.subheader("Calorie Tracker")

        # Import from calorie_tracker.py
        from calorie_tracker import load_and_train_model, grundumsatz, donut_chart

        # Load the ML model
        try:
            model, feature_columns = load_and_train_model()
        except Exception as e:
            st.error("Error loading ML model.")
            st.exception(e)
            return

        # User ID required
        user_id = st.session_state.get("user_id")
        if user_id is None:
            st.error("Please log in first.")
            return

        user = get_profile(user_id)
        if not user:
            st.error("Could not load user profile.")
            return

        # User profile
        age = user["age"]
        weight = user["weight"]
        height = user["height"]
        gender = user.get("gender", "male")
        goal = user.get("goal", "Maintain")

        # ===========================================
        # AUTO IMPORT WORKOUT FROM WORKOUT PLANNER
        # ===========================================
        workout = st.session_state.get("current_workout")

        if workout is None:
            st.info("No workout logged today. Training calories = 0.")
            training_type = "kraft"
            duration = 0
        else:
            # Extract info
            w_title = workout.get("title", "").lower()
            duration = workout.get("minutes", 0)

            kraft_keywords = ["push", "pull", "leg", "full body", "upper", "lower"]
            if any(k in w_title for k in kraft_keywords):
                training_type = "kraft"
            else:
                training_type = "kraft"  # default fallback

            st.success(
                f"Today's workout: **{workout['title']}** — {duration} minutes"
            )

        # ML INPUT
        person = {
            "Age": age,
            "Duration": duration,
            "Weight": weight,
            "Height": height,
            "Gender_Female": 1 if gender.lower() == "female" else 0,
            "Gender_Male": 1 if gender.lower() == "male" else 0,
            "Training_Type_Cardio": 1 if training_type == "cardio" else 0,
            "Training_Type_Kraft": 1 if training_type == "kraft" else 0,
        }

        person_df = pd.DataFrame([person])
        person_df = person_df.reindex(columns=feature_columns, fill_value=0)

        try:
            training_kcal = float(model.predict(person_df)[0])
        except Exception:
            training_kcal = 0

        # BMR & daily total target
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

        # Ensure meal list exists
        if "ct_meals" not in st.session_state:
            st.session_state.ct_meals = []

        # ===========================================
        # FIRST — DOUGHNUT CHARTS (ABOVE everything)
        # ===========================================
        total_cal = sum(m["calories"] for m in st.session_state.ct_meals)
        total_prot = sum(m["protein"] for m in st.session_state.ct_meals)

        st.markdown("### Daily targets")
        c1, c2 = st.columns(2)
        with c1:
            donut_chart(total_cal, target_calories, "Calories", "kcal")
        with c2:
            donut_chart(total_prot, target_protein, "Protein", "g")

        # ===========================================
        # NEXT — MEALS EATEN FROM RECIPE LOG
        # ===========================================
        st.markdown("### Meals eaten today (from recipe logging)")

        meal_log = st.session_state.meal_log
        if not meal_log:
            st.info("No meals logged today.")
        else:
            df_log_simple = pd.DataFrame(meal_log)[["date_str", "recipe_name", "calories", "protein"]]
            df_log_simple.columns = ["Date", "Meal", "Calories", "Protein (g)"]
            st.table(df_log_simple)

        # ===========================================
        # USER MANUAL MEAL ADDITION BELOW
        # ===========================================
        st.markdown("### Add additional meals (manual)")

        with st.form("ct_meal_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            meal_name = c1.text_input("Meal name", "Chicken & rice")
            meal_cal = c2.number_input("Calories", 0, 3000, 500)
            meal_prot = c3.number_input("Protein (g)", 0, 200, 30)
            submitted = st.form_submit_button("Add meal")

        if submitted:
            st.session_state.ct_meals.append({
                "meal": meal_name,
                "calories": float(meal_cal),
                "protein": float(meal_prot),
            })

        if st.button("Reset manual meals"):
            st.session_state.ct_meals = []

        if st.session_state.ct_meals:
            st.markdown("### Additional manual meals")
            st.table(pd.DataFrame(st.session_state.ct_meals))


