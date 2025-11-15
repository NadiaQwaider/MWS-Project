# utils.py
import streamlit as st
import plotly.express as px

def plot_fitness_interactive(fitness_history):
    if not fitness_history:
        st.info("No fitness history to plot yet.")
        return
    fig = px.line(x=list(range(1, len(fitness_history)+1)),
                  y=fitness_history,
                  labels={"x":"Generation", "y":"Best Accuracy"},
                  title="Fitness (Best Accuracy) over Generations")
    st.plotly_chart(fig, use_container_width=True)

def display_selected_features(features):
    if not features:
        st.warning("No features selected.")
        return
    st.markdown("### ✅ Selected Features")
    for f in features:
        st.write(f"- {f}")
