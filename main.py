import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
from game_rating_classification import EEG 
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.instance = EEG()
        
    
    def main():
        st.sidebar.title("Menu")
        # Menu seçenekleri
        page = st.sidebar.radio(
            "Select Page",
            ["Menu", "📊 PEGI Feature Importances", "📊 ESRB Feature Importances"]
        )

        # Sayfa içerikleri
        if page == "Menu":
            st.title("🎮 EEG-Based Game Rating Classification")
            st.write("""
            ### Project Description:
            This project focuses on predicting **game rating classifications (PEGI and ESRB)** using EEG-based features.
            The system uses different machine learning models (Random Forest, KNN, Naive Bayes) to classify the game ratings.
            The best performing model is selected automatically based on accuracy, and feature importance visualizations are generated.
            """)

        elif page == "🏁 PEGI Feature Importances":
            st.title("PEGI Feature Importance Visualization")
            st.write("Below is the feature importance graph for PEGI game rating classification.")
            st.image("model_results/kmeans_model_RandomForest_pegi.png", caption="Top EEG Features for PEGI Classification", use_container_width=True)

        elif page == "📊 ESRB Feature Importances":
            st.title("ESRB Feature Importance Visualization")
            st.write("Below is the feature importance graph for ESRB game rating classification.")
            st.image("model_results/kmeans_model_RandomForest_esrb.png", caption="Top EEG Features for ESRB Classification", use_container_width=True)



if __name__ == "__main__":
    x = Main()
    x.main()


