import streamlit as st

st.set_page_config(
    page_title="Flight Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    st.switch_page("pages/1_Dashboard_Overview.py")
except Exception as e:
    from src.app_helper import initialize_app_state
    initialize_app_state()
    st.title("Flight Delay Intelligence")
    st.info("Please select 'Dashboard Overview' from the sidebar menu to begin.")
