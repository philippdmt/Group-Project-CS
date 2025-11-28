import ast
import json
from fractions import Fraction
from collections import Counter
from datetime import date
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from database import get_profile

PRIMARY_COLOR = "#007A3D"

# -------------------------------------------------------
# SAFE CONVERSION
# -------------------------------------------------------
def to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except:
        return None

# -------------------------------------------------------
# ML MODEL (CALORIE TRACKER)
# -------------------------------------------------------
CSV_URL = "https://raw.githubusercontent.com/philippdmt/Protein_and_Calories/refs/heads/main/calories.csv"

def determine_training_type(hr, age):
    if hr >= 0.6 * (220 - age):
        return "Cardio"
    return "Kraft"

@st.cache_data
def load_and_train_model():
    try:
        df = pd.read_csv("calories.csv")
    except:
        df = pd.read_csv(CSV_URL)

    df["Training_Type"] = df.apply(lambda r: determine_training_type(r["Heart_Rate"], r["Age"]), axis=1)

    y = df["Calories"]
    X = df.drop(columns=["User_ID", "Heart_Rate", "Body_Temp", "Calories"])

    X = pd.get_dummies(X, columns=["Gender", "Training_Type"], drop_first=False)

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns.tolist()

def grundumsatz(age, weight, height, gender):
    if gender.lower() == "male":
        return 10*weight + 6.25*height - 5*age + 5
    return 10*weight + 6.25*height - 5*age - 161


# -------------------------------------------------------
# DONUT CHART
# -------------------------------------------------------
def donut_chart(consumed, total, title, unit):
    consumed = max(consumed, 0)
    remaining = max(total - consumed, 0)
    if total <= 0:
        total = 1

    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(
        [consumed, remaining],
        colors=[PRIMARY_COLOR, "#E0E0E0"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width":0.35, "edgecolor": "white"}
    )
    ax.set(aspect="equal")
    ax.set_title(title)
    ax.text(0, 0, f"{int(consumed)} / {int(total)} {unit}", ha="center", va="center")
    st.pyplot(fig)


# -------------------------------------------------------
# INGREDIENT PARSING + FRACTIONS
# -------------------------------------------------------
UNICODE_FRACTIONS = {
    "¼":"1/4","½":"1/2","¾":"3/4",
    "⅐":"1/7","⅑":"1/9","⅒":"1/10",
    "⅓":"1/3","⅔":"2/3",
    "⅕":"1/5","⅖":"2/5","⅗":"3/5","⅘":"4/5",
    "⅙":"1/6","⅚":"5/6",
    "⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"
}

def float_to_fraction_str(x):
    f = Fraction(x).limit_denominator(16)
    if f.denominator == 1:
        return f"{f.numerator}"
    return f"{f.numerator}/{f.denominator}"

def parse_quantity_token(tok):
    if tok in UNICODE_FRACTIONS:
        tok = UNICODE_FRACTIONS[tok]
    try:
        return float(Fraction(tok))
    except:
        return None

def split_quantity_from_line(line):
    if not line:
        return None, line
    tokens = line.split()
    qty_tokens=[]
    rest=[]
    for i,t in enumerate(tokens):
        val = parse_quantity_token(t)
        if val is not None and not rest:
            qty_tokens.append(t)
        else:
            rest = tokens[i:]
            break
    if not qty_tokens:
        return None, line
    qty = sum(parse_quantity_token(q) for q in qty_tokens)
    rest=" ".join(rest)
    return qty, rest

def scale_ingredient_lines(lines, factor):
    out=[]
    for line in lines:
        qty, rest = split_quantity_from_line(line)
        if qty is None:
            out.append(line)
        else:
            new = qty * factor
            out.append(f"{float_to_fraction_str(new)} {rest}")
    return out


# -------------------------------------------------------
# LOAD RECIPES
# -------------------------------------------------------
DATA_URL = (
    "https://huggingface.co/datasets/datahiveai/recipes-with-nutrition/"
    "resolve/main/recipes-with-nutrition.csv"
)

@st.cache_data(show_spinner=True)
def load_recipes():
    df = pd.read_csv(DATA_URL)

    def get_nutr(json_str, k):
        try:
            d = json.loads(json_str)
            return float(d[k]["quantity"]) if k in d else None
        except:
            return None

    df["protein_total"] = df["total_nutrients"].apply(lambda x: get_nutr(x,"PROCNT"))
    df["fat_total"]     = df["total_nutrients"].apply(lambda x: get_nutr(x,"FAT"))
    df["carbs_total"]   = df["total_nutrients"].apply(lambda x: get_nutr(x,"CHOCDF"))

    df = df.dropna(subset=["protein_total","fat_total","carbs_total","calories","servings"])

    df["protein_g"] = df["protein_total"] / df["servings"]
    df["fat_g"]     = df["fat_total"]     / df["servings"]
    df["carbs_g"]   = df["carbs_total"]   / df["servings"]
    df["calories_per_serving"] = df["calories"] / df["servings"]

    # Ingredients list
    def parse_ing(x):
        try:
            v = ast.literal_eval(str(x))
            if isinstance(v,list):
                out=[]
                for el in v:
                    if isinstance(el,dict) and "text" in el:
                        out.append(el["text"].lower())
                    else:
                        out.append(str(el).lower())
                return out
        except:
            pass
        return [p.strip().lower() for p in str(x).split(",")]

    df["ingredients_list"] = df["ingredients"].apply(parse_ing)

    # Ingredient lines
    def parse_lines(x):
        try:
            v = ast.literal_eval(str(x))
            if isinstance(v,list):
                return [str(e) for e in v]
        except:
            pass
        return [str(x)]

    df["ingredient_lines_parsed"] = df["ingredient_lines"].apply(parse_lines)
    df["ingredient_lines_per_serving"] = df.apply(
        lambda r: scale_ingredient_lines(r["ingredient_lines_parsed"], 1/float(r["servings"])),
        axis=1
    )

    # meal_type fallback
    if "meal_type" not in df.columns:
        df["meal_type"] = "unknown"
    df["meal_type"] = df["meal_type"].fillna("unknown")

    return df


# -------------------------------------------------------
# FILTERING + MEAL PICKING
# -------------------------------------------------------
def filter_recipes(df, diet_pref, allergies):
    diet_pref = (diet_pref or "No preference").lower()
    allergies = [a.strip().lower() for a in allergies if a]

    base = df.copy()

    if diet_pref == "vegan":
        base = base[base["diet_labels"].str.contains("vegan",case=False,na=False)]
    elif diet_pref == "vegetarian":
        base = base[base["diet_labels"].str.contains("vegetarian",case=False,na=False)]

    if allergies:
        base = base[base["ingredients_list"].apply(
            lambda ing: not any(a in x for a in allergies for x in ing)
        )]

    return base

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
        total=0
        for ing in row.get("ingredients_list",[]):
            total += self.liked.get(ing,0)
            total -= self.disliked.get(ing,0)
        return total

def pick_meal(df, meal_type, target, pref_model):
    subset = df[df["meal_type"].astype(str).str.contains(meal_type,case=False,na=False)]
    if subset.empty:
        return None

    subset = subset.copy()
    subset["cal_diff"] = (subset["calories_per_serving"] - target).abs()

    if pref_model:
        subset["score"] = subset.apply(pref_model.score, axis=1)
        subset = subset.sort_values(["score","cal_diff"], ascending=[False,True])
    else:
        subset = subset.sort_values("cal_diff")

    return subset.sample(1).iloc[0]


# -------------------------------------------------------
# SHOW RECIPE CARD
# -------------------------------------------------------
def show_recipe_card(row, key_prefix, pref_model):
    if row is None:
        st.write("No recipe found.")
        return

    if isinstance(row, tuple):  # safety fix
        row = row[0]

    eaten = any(m["meal"] == row["recipe_name"] for m in st.session_state.meal_log)

    with st.container():
        col1, col2 = st.columns([1,3])

        with col1:
            img = row.get("image_url","")
            if isinstance(img,str) and img.strip():
                st.image(img, width=220)
            else:
                st.write("No image")

        with col2:
            st.subheader(row["recipe_name"])

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Calories", f"{row['calories_per_serving']:.0f}")
            c2.metric("Protein", f"{row['protein_g']:.1f}g")
            c3.metric("Carbs",   f"{row['carbs_g']:.1f}g")
            c4.metric("Fat",     f"{row['fat_g']:.1f}g")

            st.markdown("**Ingredients (per serving):**")
            for l in row["ingredient_lines_per_serving"]:
                st.markdown(f"- {l}")

            st.markdown("---")

            if not eaten:
                if st.button("I ate this", key=f"eat_{key_prefix}"):
                    st.session_state.meal_log.append({
                        "date_str": date.today().strftime("%d/%m/%Y"),
                        "meal": row["recipe_name"],
                        "calories": row["calories_per_serving"],
                        "protein": row["protein_g"]
                    })
                    st.success("Added to meal log.")
                if st.button("I don't like this", key=f"skip_{key_prefix}"):
                    pref_model.update_with_rating(row, -1)
                return

            if st.button("I liked it", key=f"like_{key_prefix}"):
                pref_model.update_with_rating(row, +1)
            if st.button("Favourite", key=f"fav_{key_prefix}"):
                st.session_state.favourite_recipes.add(row.name)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    # login
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in first.")
        return

    user = get_profile(st.session_state.user_id)

    # safe profile conversion
    age    = to_float(user.get("age"))
    weight = to_float(user.get("weight"))
    height = to_float(user.get("height"))
    gender = user.get("gender","Male")
    goal   = user.get("goal","Maintain")
    allergies = (user.get("allergies") or "").split(",")
    diet_pref = user.get("diet_preferences","No preference")

    if age is None or weight is None or height is None:
        st.error("Please complete your profile (age, weight, height).")
        return

    # session state init
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []
    st.session_state.pref_model = UserPreferenceModel()
    if "favourite_recipes" not in st.session_state:
        st.session_state.favourite_recipes = set()
    if "daily_plan" not in st.session_state:
        st.session_state.daily_plan = None

    # ML model
    model, feature_cols = load_and_train_model()

    # workout data
    wo = st.session_state.get("current_workout")
    if wo:
        minutes = wo["minutes"]
        title = wo["title"].lower()
        training_type = "Kraft" if any(k in title for k in ["push","pull","leg","upper","lower","full"]) else "Cardio"
    else:
        minutes=0
        training_type="Kraft"

    # BMR
    bmr = grundumsatz(age, weight, height, gender)

    person = {
        "Age": age,
        "Duration": minutes,
        "Weight": weight,
        "Height": height,
        "Gender_Female": 1 if gender.lower()=="female" else 0,
        "Gender_Male":   1 if gender.lower()=="male" else 0,
        "Training_Type_Cardio": 1 if training_type=="Cardio" else 0,
        "Training_Type_Kraft":  1 if training_type=="Kraft" else 0,
    }

    dfp = pd.DataFrame([person]).reindex(columns=feature_cols, fill_value=0)
    dfp = dfp.fillna(0)

    try:
        training_kcal = float(model.predict(dfp)[0]) if wo else 0
    except:
        training_kcal=0

    if goal.lower()=="bulk":
        target_cal = bmr + training_kcal + 300
        prot_factor=2.0
    elif goal.lower()=="cut":
        target_cal = bmr + training_kcal - 300
        prot_factor=2.2
    else:
        target_cal = bmr + training_kcal
        prot_factor=1.6

    target_cal = max(target_cal,1200)
    target_prot = prot_factor * weight

    # load recipes
    recipes = load_recipes()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Daily Overview",
        "Daily Recipe Plan",
        "Search Recipes",
        "Favourites"
    ])

    # -----------------------------------
    # TAB 1: DAILY OVERVIEW
    # -----------------------------------
    with tab1:
        st.subheader("Daily Overview")

        c1,c2 = st.columns(2)
        with c1:
            donut_chart(sum(m["calories"] for m in st.session_state.meal_log), target_cal, "Calories", "kcal")
        with c2:
            donut_chart(sum(m["protein"] for m in st.session_state.meal_log), target_prot, "Protein", "g")

        st.markdown("### Log a meal")
        with st.form("meal_form"):
            n1,n2,n3 = st.columns([2,1,1])
            name = n1.text_input("Meal", "Chicken & Rice")
            cal = n2.number_input("Calories", 0, 3000, 500)
            prot = n3.number_input("Protein", 0, 200, 30)
            s = st.form_submit_button("Add")
        if s:
            st.session_state.meal_log.append({
                "date_str": date.today().strftime("%d/%m/%Y"),
                "meal": name,
                "calories": float(cal),
                "protein": float(prot)
            })

        if st.button("Reset meals"):
            st.session_state.meal_log = []

        st.markdown("### Meals today")
        if st.session_state.meal_log:
            st.table(pd.DataFrame(st.session_state.meal_log))
        else:
            st.info("No meals yet.")

    # -----------------------------------
    # TAB 2: DAILY PLAN
    # -----------------------------------
    with tab2:
        st.subheader("Daily Recipe Plan")

        if st.button("Generate daily plan"):
            user_df = filter_recipes(recipes, diet_pref, allergies)
            plan = {
                "Breakfast": pick_meal(user_df, "breakfast", target_cal*0.25, st.session_state.pref_model),
                "Lunch":     pick_meal(user_df, "lunch",     target_cal*0.40, st.session_state.pref_model),
                "Dinner":    pick_meal(user_df, "dinner",    target_cal*0.35, st.session_state.pref_model),
            }
            st.session_state.daily_plan = plan

        plan = st.session_state.daily_plan
        if not plan:
            st.info("Click the button to create your plan.")
        else:
            for meal_name, row in plan.items():
                st.markdown(f"### {meal_name}")
                show_recipe_card(row, meal_name, st.session_state.pref_model)
                st.markdown("---")

    # -----------------------------------
    # TAB 3: SEARCH RECIPES
    # -----------------------------------
    with tab3:
        st.subheader("Search Recipes")

        c1,c2,c3 = st.columns(3)
        inc = c1.text_input("Must include")
        exc = c2.text_input("Exclude")
        meal_t = c3.selectbox("Meal type", ["all","breakfast","lunch","dinner"])

        inc_l = [i.strip().lower() for i in inc.split(",") if i.strip()]
        exc_l = [i.strip().lower() for i in exc.split(",") if i.strip()]
        max_cal = st.number_input("Max calories",0,3000,800)

        if st.button("Search"):
            df = filter_recipes(recipes, diet_pref, allergies)
            if meal_t!="all":
                df = df[df["meal_type"].str.contains(meal_t,case=False,na=False)]
            if inc_l:
                df = df[df["ingredients_list"].apply(
                    lambda ing: all(any(i in x for x in ing) for i in inc_l)
                )]
            if exc_l:
                df = df[df["ingredients_list"].apply(
                    lambda ing: not any(any(e in x for x in ing) for e in exc_l)
                )]
            df = df[df["calories_per_serving"] <= max_cal]

            if df.empty:
                st.warning("No recipes.")
            else:
                for idx,row in df.head(20).iterrows():
                    show_recipe_card(row, f"search_{idx}", st.session_state.pref_model)
                    st.markdown("---")

    # -----------------------------------
    # TAB 4: FAVOURITES
    # -----------------------------------
    with tab4:
        st.subheader("Favourites")

        favs = st.session_state.favourite_recipes
        if not favs:
            st.info("No favourites yet.")
        else:
            for idx in favs.copy():
                if idx not in recipes.index:
                    favs.discard(idx)
                    continue
                row = recipes.loc[idx]
                show_recipe_card(row, f"fav_{idx}", st.session_state.pref_model)
                st.markdown("---")
