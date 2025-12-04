# workout_calendar.py
import calendar
import datetime
import streamlit as st
from streamlit import session_state as state
PRIMARY_COLOR = "#007A3D"
# Weekday names + index matching Python datetime.weekday()
WEEKDAYS = [
    ("Monday", 0),
    ("Tuesday", 1),
    ("Wednesday", 2),
    ("Thursday", 3),
    ("Friday", 4),
    ("Saturday", 5),
    ("Sunday", 6),
]
# Workout categories selectable by user
WORKOUT_TYPES = [
    "Push day",
    "Pull day",
    "Leg day",
    "Full body",
    "Lower body",
    "Cardio",
]
def _init_state():
    """
    Ensure persistent variables exist inside Streamlit session_state.
    This keeps data while user interacts with the app.
    """
    # Stores workout logs keyed by date string YYYY-MM-DD
    # Example:
    #  { "2025-12-02": {"minutes": 45, "type": "Push day"} }
    if "workout_logs" not in state:
        state.workout_logs = {}
    # Stores currently selected calendar date
    if "selected_date" not in state:
        state.selected_date = datetime.date.today()
def _render_month(year: int, month: int):
    """
    Render an interactive month calendar.
    Each date is clickable -> selecting a date to log/edit workouts.
    Logged days show brief summary below button.
    """
    # Create a calendar object starting week on Monday
    cal = calendar.Calendar(firstweekday=0)
    # Display month heading (e.g., "December 2025")
    month_name = calendar.month_name[month]
    st.markdown(f"#### {month_name} {year}")
    # Display weekday header row
    header_cols = st.columns(7)
    for col, (name, _) in zip(header_cols, WEEKDAYS):
        col.markdown(
            f"<div style='text-align:center; font-size:13px; "
            f"color:#555; font-weight:600;'>{name[:2]}</div>",
            unsafe_allow_html=True,
        )
    # Loop through weeks and dates in the month
    for week in cal.monthdatescalendar(year, month):
        cols = st.columns(7)
        for col, date_obj in zip(cols, week):
            # Only display real days belonging to this month
            if date_obj.month != month:
                with col:
                    st.write("")  # empty cell placeholder
                continue
            date_str = date_obj.isoformat()
            log = state.workout_logs.get(date_str)
            is_selected = (state.selected_date == date_obj)
            day_label = str(date_obj.day)
            with col:
                # Show marker if this is the selected day
                if is_selected:
                    st.markdown(
                        f"<div style='text-align:center; font-size:11px; "
                        f"color:{PRIMARY_COLOR};'>Selected</div>",
                        unsafe_allow_html=True,
                    )
                # Create interactive button for the date
                if st.button(
                    day_label,
                    key=f"day-btn-{date_str}",
                    use_container_width=True,
                ):
                    state.selected_date = date_obj  # update selected day
                # If workout exists for this day, show a summary
                if log:
                    minutes = log.get("minutes")
                    wtype = log.get("type")
                    summary = f"{minutes} min – {wtype}"
                    st.caption(summary)
def main():
    """Main UI function to render the workout calendar tab."""
    _init_state()  # ensure state variables exist
    st.subheader("Workout log calendar")
    st.caption(
        "Log your past workouts in a calendar view. "
        "Click on a date to add or edit the workout for that day."
    )
    st.info(
        "You always see the current month plus the previous two months (3 months of past workouts)."
    )
    st.divider()
    # Determine the date index to count months backwards
    today = datetime.date.today()
    base_index = today.year * 12 + (today.month - 1)
    # Always show 3 months — current month + previous 2
    months_to_show = 3
    # Render the calendar for the last 3 months
    for offset in range(months_to_show):
        idx = base_index - offset
        year = idx // 12
        month = (idx % 12) + 1
        _render_month(year, month)
        st.write("")  # spacing between month sections
    st.divider()
    # Workout logging panel
    selected_date: datetime.date = state.selected_date
    selected_str = selected_date.strftime("%A, %d %B %Y")
    st.markdown(f"### Log workout for {selected_str}")
    date_key = selected_date.isoformat()
    # Load existing entry if exists; otherwise default values
    existing_log = state.workout_logs.get(date_key, {})
    existing_minutes = existing_log.get("minutes", 0)
    existing_type = existing_log.get("type", WORKOUT_TYPES[0])
    # Two columns — one input for minutes, one dropdown for workout type
    col_minutes, col_type = st.columns([1, 1])
    with col_minutes:
        minutes = st.number_input(
            "Workout length (minutes)",
            min_value=0,
            max_value=600,
            step=5,
            value=int(existing_minutes),
            help="How long did you train on this day?",
            key="minutes_input",
        )
    with col_type:
        # Ensure saved type still exists in case the list changes
        if existing_type not in WORKOUT_TYPES:
            existing_type = WORKOUT_TYPES[0]
        workout_type = st.selectbox(
            "Workout type",
            options=WORKOUT_TYPES,
            index=WORKOUT_TYPES.index(existing_type),
            help="What kind of workout did you do?",
            key="workout_type_select",
        )
    # Save workout data to state
    if st.button("Save workout", type="primary"):
        state.workout_logs[date_key] = {
            "minutes": int(minutes),
            "type": workout_type,
        }
        st.success(f"Saved workout for {selected_str}.")
    # Optional: Show most recent logged workouts below
    if state.workout_logs:
        st.markdown("#### Recent logged workouts")
        # Sort log entries by latest first
        items = sorted(state.workout_logs.items(), key=lambda x: x[0], reverse=True)
        for d_str, log in items[:10]:  # show last 10 entries
            d_obj = datetime.date.fromisoformat(d_str)
            label = d_obj.strftime("%d %b %Y (%a)")
            st.write(f"- **{label}**: {log['minutes']} min – {log['type']}")
