import streamlit as st

# Page setup
dashpage = st.Page(
    "views/prediction_dashboard.py",
    title="Prediction Page",
    icon=":material/bar_chart:",
    default=True,
)


# Navigation setup
pg = st.navigation(
    {
        "Toolbar": [dashpage],
    }
)

# Set logo (ensure assets/midhun.png exists)
st.logo("assets/iti_logo.jpeg")

# Run navigation
pg.run()