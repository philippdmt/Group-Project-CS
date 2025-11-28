import ast
import json
from fractions import Fraction
from collections import Counter
from typing import List, Optional
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px  # currently unused, but kept for later plots


# =============================================================================
# CONFIG
# =============================================================================

# Remote CSV from HuggingFace (no local path needed)
DATA_URL = (
    "https://huggingface.co/datasets/datahiveai/recipes-with-nutrition"
    "/resolve/main/recipes-with-nutrition.csv"
)

HIGH_PROTEIN_MIN = 25
MAX_CALORIES_PER_SERVING = 800

DAILY_CALORIES_PLACEHOLDER = 2000

PROTEIN_RATIO = 0.30
CARB_RATIO = 0.40
FAT_RATIO = 0.30


# =============================================================================
# SIMPLE PREFERENCE MODEL (PLATZHALTER FÜR ML)
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
# INGREDIENT PARSER – Text → Zahl → teilen → Bruch
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

    if "-" in token and not token.startswith("-"):
        try:
            a, b = token.split("-")
            return (float(Fraction(a)) + float(Fraction(b))) / 2
        except Exception:
            pass

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
        frac = float_to_fraction_str(new_qty)
        scaled.append(f"{frac} {rest}".strip())
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
    if pd.isna(x):
        return []
    s = str(x)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            if all(isinstance(v, dict) and "text" in v for v in val):
                return [str(v["text"]).strip().lower() for v in val]
            return [str(v).strip().lower() for v in val]
    except Exception:
        pass
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def parse_ingredient_lines_for_display(x):
    if pd.isna(x):
        return []
    s = str(x)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
    except Exception:
        pass
    return [p.strip() for p in s.split(",") if p.strip()]


@st.cache_data(show_spinner=True)
def load_and_prepare_data(path_or_url: str) -> pd.DataFrame:
    """Load dataset (CSV) from a local path or URL and prepare fitness recipes."""
    df = pd.read_csv(path_or_url)

    df["protein_g_total"] = df["total_nutrients"].apply(lambda x: get_nutrient(x, "PROCNT"))
    df["fat_g_total"]     = df["total_nutrients"].apply(lambda x: get_nutrient(x, "FAT"))
    df["carbs_g_total"]   = df["total_nutrients"].apply(lambda x: get_nutrient(x, "CHOCDF"))

    df = df.dropna(subset=["protein_g_total", "fat_g_total", "carbs_g_total", "calories", "servings"])

    df["protein_g_total"] = df["protein_g_total"].astype(float)
    df["fat_g_total"]     = df["fat_g_total"].astype(float)
    df["carbs_g_total"]   = df["carbs_g_total"].astype(float)
    df["calories"]        = df["calories"].astype(float)

    df["servings"] = pd.to_numeric(df["servings"], errors="coerce").fillna(1)
    df.loc[df["servings"] <= 0, "servings"] = 1

    df["calories_per_serving"] = df["calories"] / df["servings"]
    df["protein_g"] = df["protein_g_total"] / df["servings"]
    df["fat_g"]     = df["fat_g_total"] / df["servings"]
    df["carbs_g"]   = df["carbs_g_total"] / df["servings"]

    fitness_df = df[
        (df["protein_g"] >= HIGH_PROTEIN_MIN) &
        (df["calories_per_serving"] <= MAX_CALORIES_PER_SERVING)
    ].copy()

    fitness_df["ingredients_list"] = fitness_df["ingredients"].apply(parse_ingredients_for_allergy)
    fitness_df["ingredient_lines_parsed"] = fitness_df["ingredient_lines"].apply(parse_ingredient_lines_for_display)

    def per_serving_lines(row):
        factor = 1.0 / float(row["servings"])
        return scale_ingredient_lines(row["ingredient_lines_parsed"], factor)

    fitness_df["ingredient_lines_per_serving"] = fitness_df.apply(per_serving_lines, axis=1)

    return fitness_df


# =============================================================================
# FILTER / SEARCH / PLAN
# =============================================================================

def filter_by_preferences(df: pd.DataFrame, diet_pref: str, allergies: List[str]) -> pd.DataFrame:
    diet_pref = (diet_pref or "omnivore").lower()
    allergies = [a.strip().lower() for a in allergies if a.strip()]

    res = df.copy()
    if diet_pref == "vegan":
        res = res[res["diet_labels"].astype(str).str.contains("vegan", case=False, na=False)]
    elif diet_pref == "vegetarian":
        res = res[
            res["diet_labels"].astype(str).str.contains("vegetarian", case=False, na=False) |
            res["diet_labels"].astype(str).str.contains("vegan", case=False, na=False)
        ]

    if allergies:
        res = res[res["ingredients_list"].apply(
            lambda ing: not any(any(a in x for x in ing) for a in allergies)
        )]

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

    include_ingredients = [i.strip().lower() for i in include_ingredients if i.strip()]
    exclude_ingredients = [i.strip().lower() for i in exclude_ingredients if i.strip()]

    if meal_type != "all":
        base = base[base["meal_type"].astype(str).str.contains(meal_type, case=False, na=False)]

    if max_calories:
        base = base[base["calories_per_serving"] <= max_calories]

    if include_ingredients:
        base = base[base["ingredients_list"].apply(
            lambda ing: all(any(inc in x for x in ing) for inc in include_ingredients)
        )]

    if exclude_ingredients:
        base = base[base["ingredients_list"].apply(
            lambda ing: not any(any(exc in x for x in ing) for exc in exclude_ingredients)
        )]

    if pref_model is not None and not base.empty:
        base = base.copy()
        base["score"] = base.apply(pref_model.score_recipe, axis=1)
        base = base.sort_values("score", ascending=False)

    return base


def pick_meal(
    df: pd.DataFrame,
    meal_type: str,
    target_cal: float,
    training_goal: str,
    pref_model: Optional[UserPreferenceModel],
) -> Optional[pd.Series]:

    base = df[df["meal_type"].astype(str).str.contains(meal_type, case=False, na=False)]
    if base.empty:
        return None

    base = base.copy()
    base["cal_diff"] = (base["calories_per_serving"] - target_cal).abs()

    g = (training_goal or "").lower()
    if g == "strength":
        base = base.sort_values(["protein_g", "cal_diff"], ascending=[False, True])
    elif g == "endurance":
        base = base.sort_values(["carbs_g", "cal_diff"], ascending=[False, True])
    else:
        base = base.sort_values("cal_diff", ascending=True)

    if pref_model is not None:
        base["score"] = base.apply(pref_model.score_recipe, axis=1)
        base = base.sort_values(["score", "cal_diff"], ascending=[False, True])

    top_n = min(20, len(base))
    return base.head(top_n).sample(1).iloc[0]


def recommend_daily_plan(
    df: pd.DataFrame,
    daily_calories: float,
    training_goal: str,
    diet_pref: str,
    allergies: List[str],
    pref_model: Optional[UserPreferenceModel],
):
    user_df = filter_by_preferences(df, diet_pref, allergies)

    breakfast_cal = daily_calories * 0.25
    lunch_cal     = daily_calories * 0.40
    dinner_cal    = daily_calories * 0.35

    breakfast = pick_meal(user_df, "breakfast", breakfast_cal, training_goal, pref_model)
    lunch     = pick_meal(user_df, "lunch",     lunch_cal,     training_goal, pref_model)
    dinner    = pick_meal(user_df, "dinner",    dinner_cal,    training_goal, pref_model)

    return {
        "Breakfast": (breakfast, breakfast_cal),
        "Lunch":     (lunch,     lunch_cal),
        "Dinner":    (dinner,    dinner_cal),
    }


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if "pref_model" not in st.session_state:
        st.session_state.pref_model = UserPreferenceModel()
    if "consumed" not in st.session_state:
        st.session_state.consumed = {"cal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    if "favourite_recipes" not in st.session_state:
        st.session_state.favourite_recipes = set()  # indices
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []  # list of dicts
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "daily_plan" not in st.session_state:
        st.session_state.daily_plan = None
    if "eaten_today" not in st.session_state:
        st.session_state.eaten_today = set()  # recipe_name strings
    if "rating_stage" not in st.session_state:
        st.session_state.rating_stage = {}  # recipe_name -> stage


def log_meal(row: pd.Series, meal_name: str):
    entry = {
        "date_str":   date.today().strftime("%d/%m/%Y"),
        "recipe_name": row["recipe_name"],
        "meal_name":   meal_name,
        "calories":    row["calories_per_serving"],
        "protein":     row["protein_g"],
        "carbs":       row["carbs_g"],
        "fat":         row["fat_g"],
    }
    st.session_state.meal_log.append(entry)

    st.session_state.consumed["cal"]     += entry["calories"]
    st.session_state.consumed["protein"] += entry["protein"]
    st.session_state.consumed["carbs"]   += entry["carbs"]
    st.session_state.consumed["fat"]     += entry["fat"]

    st.session_state.eaten_today.add(row["recipe_name"])


# =============================================================================
# RECIPE CARD
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
    """
    mode:
      - "default": normal (Suggested / Search)
      - "favourite": Favourite recipes tab (nur Anzeigen + Entfernen)
    """
    if row is None:
        st.write("No suitable recipe found.")
        return

    recipe_name = row["recipe_name"]
    eaten = recipe_name in st.session_state.eaten_today
    rating_stage = st.session_state.rating_stage.get(recipe_name, "none")

    container = st.container()
    with container:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            img_url = row.get("image_url", "")
            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, width=240)
            else:
                st.write("No image available.")

        with col_right:
            st.subheader(recipe_name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{row['calories_per_serving']:.0f}")
            c2.metric("Protein",  f"{row['protein_g']:.1f} g")
            c3.metric("Carbs",    f"{row['carbs_g']:.1f} g")
            c4.metric("Fat",      f"{row['fat_g']:.1f} g")

            st.markdown("**Ingredients (per serving)**")
            for line in row.get("ingredient_lines_per_serving", []):
                st.markdown(f"- {line}")

        st.markdown("---")

        # Button-Leiste immer gleich: Link | Button 1 | Button 2
        b0, b1, b2 = st.columns(3)

        # Link (wenn vorhanden)
        with b0:
            if pd.notna(row.get("url", None)) and str(row["url"]).strip():
                st.markdown(f"[Go to recipe]({row['url']})")

        # ----------------------- Favourite-Mode -----------------------
        if mode == "favourite":
            with b1:
                if st.button("Remove from favourite recipes", key=f"remove_{key_prefix}"):
                    st.session_state.favourite_recipes.discard(row.name)
            st.markdown("---")
            return

        # ----------------------- Default-Mode -------------------------

        # 1) Noch nicht gegessen → Eat / Skip
        if not eaten:
            with b1:
                if st.button("I have eaten this", key=f"eat_{key_prefix}"):
                    log_meal(row, meal_name)
                    st.session_state.rating_stage[recipe_name] = "none"
            with b2:
                if st.button("I don't like to eat this meal", key=f"skip_{key_prefix}"):
                    if (
                        df is not None
                        and profile is not None
                        and meal_target_calories is not None
                        and st.session_state.daily_plan is not None
                    ):
                        user_df = filter_by_preferences(df, profile["diet_pref"], profile["allergies"])
                        # Meal-Type ableiten
                        mt = str(row.get("meal_type", "")).lower()
                        if not mt:
                            mt = meal_name.lower()
                        new_row = pick_meal(
                            user_df,
                            meal_type=mt,
                            target_cal=meal_target_calories,
                            training_goal=profile["training_goal"],
                            pref_model=pref_model,
                        )
                        if new_row is not None and meal_name in st.session_state.daily_plan:
                            st.session_state.daily_plan[meal_name] = (new_row, meal_target_calories)
            st.markdown("---")
            return

        # 2) Ab hier: gegessen → Rating / Favourites

        # Stage 1: Liked / Didn't like
        if rating_stage == "none":
            with b1:
                if st.button("I liked this meal", key=f"like_{key_prefix}"):
                    if pref_model is not None:
                        pref_model.update_with_rating(row, +1)
                    st.session_state.rating_stage[recipe_name] = "liked"
            with b2:
                if st.button("I didn't like this meal", key=f"dislike_{key_prefix}"):
                    if pref_model is not None:
                        pref_model.update_with_rating(row, -1)
                    st.session_state.rating_stage[recipe_name] = "disliked"

        # Stage 2: nach liked → Save / Don't save
        elif rating_stage == "liked":
            with b1:
                if st.button("Save in favourites", key=f"fav_{key_prefix}"):
                    st.session_state.favourite_recipes.add(row.name)
                    st.session_state.rating_stage[recipe_name] = "liked_saved"
            with b2:
                if st.button("Don't save in favourites", key=f"nofav_{key_prefix}"):
                    st.session_state.rating_stage[recipe_name] = "liked_nosave"

        # andere Stages (disliked/liked_saved/liked_nosave) → keine Buttons mehr
        st.markdown("---")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """
    Embedded entry point for the Nutrition Advisory module.

    NOTE: No st.set_page_config() here – the main app handles layout & title.
    """
    init_session_state()

    st.subheader("Nutrition Advisory")
    st.caption("High-protein recipe suggestions and daily plan powered by Pumpfessor Joe 🥗💪")

    profile = st.session_state.user_profile

    # ------------------------------------------------------ ACCOUNT / PROFILE
    if profile is None:
        st.subheader("Create your nutrition profile")

        training_split_options = [
            "Push", "Pull", "Legs", "Upper Body", "Lower Body", "Full Body", "Cardio", "Rest"
        ]
        diet_options = [
            "omnivore", "vegetarian", "vegan", "pescetarian",
            "keto", "low carb", "paleo", "mediterranean", "flexitarian"
        ]

        with st.form("account_form"):
            username = st.text_input("Username")
            training_split = st.selectbox("Training type", training_split_options)
            training_goal = st.selectbox("Training goal", ["strength", "endurance", "balanced"])
            diet_pref = st.selectbox("Diet preference", diet_options)
            allergies_text = st.text_input("Allergies (comma-separated)")
            submit = st.form_submit_button("Create account")

        if submit and username.strip():
            allergies = [a.strip() for a in allergies_text.split(",") if a.strip()]
            st.session_state.user_profile = {
                "username": username.strip(),
                "training_split": training_split,
                "training_goal": training_goal,
                "diet_pref": diet_pref,
                "allergies": allergies,
                "daily_calories": float(DAILY_CALORIES_PLACEHOLDER),
            }
            profile = st.session_state.user_profile

        if profile is None:
            # Formular ist schon gerendert, hier einfach abbrechen
            return

    # ------------------------------------------------------ MAIN UI
    st.write(f"Logged in as **{profile['username']}**")

    try:
        df = load_and_prepare_data(DATA_URL)
    except Exception as e:
        st.error(
            "Error loading recipe data from the online CSV.\n\n"
            "Check your internet connection. If the problem persists, "
            "we may need to update the CSV URL in `nutrition_advisory.py`."
        )
        st.exception(e)
        return

    tab_suggested, tab_search, tab_fav, tab_log = st.tabs(
        ["Suggested recipes", "Search recipes", "Favourite recipes", "Meals eaten"]
    )

    # --------------------------- Suggested recipes
    with tab_suggested:
        st.subheader("Suggested recipes for today")
        if st.button("Generate daily plan"):
            plan = recommend_daily_plan(
                df,
                profile["daily_calories"],
                profile["training_goal"],
                profile["diet_pref"],
                profile["allergies"],
                st.session_state.pref_model,
            )
            st.session_state.daily_plan = plan

        plan = st.session_state.daily_plan
        if plan is None:
            st.info("Click 'Generate daily plan' to get recommendations.")
        else:
            for meal_name, (row, target_cal) in plan.items():
                st.markdown(f"### {meal_name}")
                if row is None:
                    st.write("No suitable recipe found.")
                else:
                    show_recipe_card(
                        row,
                        key_prefix=f"plan_{meal_name}",
                        meal_name=meal_name,
                        mode="default",
                        df=df,
                        profile=profile,
                        pref_model=st.session_state.pref_model,
                        meal_target_calories=target_cal,
                    )

    # --------------------------- Search recipes
    with tab_search:
        st.subheader("Search recipes")

        col1, col2, col3 = st.columns(3)
        with col1:
            include_text = st.text_input("Must include ingredients (comma separated)")
        with col2:
            exclude_text = st.text_input("Exclude ingredients (comma separated)")
        with col3:
            meal_type = st.selectbox("Meal type", ["all", "breakfast", "lunch", "dinner"])

        max_cal = st.number_input(
            "Max calories per serving",
            min_value=0,
            max_value=3000,
            value=MAX_CALORIES_PER_SERVING,
        )

        include_ingredients = [x.strip() for x in include_text.split(",") if x.strip()]
        exclude_ingredients = [x.strip() for x in exclude_text.split(",") if x.strip()]

        if st.button("Search"):
            results = search_recipes(
                df,
                include_ingredients,
                exclude_ingredients,
                meal_type,
                max_cal,
                profile["diet_pref"],
                profile["allergies"],
                st.session_state.pref_model,
            )
            st.session_state.search_results = results

        results = st.session_state.get("search_results", None)
        if results is None:
            st.info("Set filters and click 'Search'.")
        elif results.empty:
            st.warning("No recipes matched your filters.")
        else:
            st.write(f"Found {len(results)} recipes (showing first 20).")
            for idx, row in results.head(20).iterrows():
                show_recipe_card(
                    row,
                    key_prefix=f"search_{idx}",
                    meal_name="Search",
                    mode="default",
                    df=None,
                    profile=None,
                    pref_model=st.session_state.pref_model,
                    meal_target_calories=None,
                )

    # --------------------------- Favourite recipes
    with tab_fav:
        st.subheader("Favourite recipes")

        fav_indices = list(st.session_state.favourite_recipes)
        if not fav_indices:
            st.write("You have no favourite recipes yet.")
        else:
            for idx in fav_indices:
                if idx not in df.index:
                    continue
                row = df.loc[idx]
                show_recipe_card(
                    row,
                    key_prefix=f"fav_{idx}",
                    meal_name="Favourite",
                    mode="favourite",
                )

    # --------------------------- Meals eaten
    with tab_log:
        st.subheader("Meals eaten")

        if not st.session_state.meal_log:
            st.write("No meals recorded yet.")
        else:
            df_log = pd.DataFrame(st.session_state.meal_log)
            df_log = df_log[["date_str", "recipe_name", "meal_name", "calories", "protein", "carbs", "fat"]]
            df_log.columns = [
                "Date",
                "Meal",
                "Type of meal",
                "Calories",
                "Protein (g)",
                "Carbs (g)",
                "Fat (g)",
            ]
            st.dataframe(df_log, use_container_width=True)


if __name__ == "__main__":
    main()
