
# 🎮 EEG-Based Game Rating Classification (PEGI & ESRB)

This project focuses on predicting **game rating classifications** (PEGI and ESRB) using **EEG-based features**.  
Machine learning models such as **Random Forest**, **K-Nearest Neighbors (KNN)**, and **Naive Bayes** are evaluated to select the best-performing algorithm for each classification task.  
The project also provides **feature importance visualizations** to highlight the most influential EEG features in the rating predictions.

---

## 📌 Project Overview

- 🎯 **Game Rating Classification:** Predicts PEGI and ESRB ratings using EEG signals.
- 🏆 **Model Selection:** Compares Random Forest, KNN, and Naive Bayes models; selects the best one automatically based on accuracy.
- 📊 **Feature Importance Analysis:** Visualizes the top EEG features that contribute most to the predictions.
- 🤖 **Outlier Detection:** (Currently placeholder, can be extended for anomaly detection.)

---

## ⚙️ Technologies Used

| Technology                | Purpose                                               |
|----------------------------|-------------------------------------------------------|
| **Python 3.x**             | Core programming language for all scripts            |
| **Scikit-learn**           | Machine learning models (Random Forest, KNN, Naive Bayes), training, evaluation, feature importance |
| **Pandas**                 | Data preprocessing, Excel reading, DataFrame operations |
| **NumPy**                  | Numerical operations and array handling              |
| **Matplotlib**             | Visualizations (feature importance plots, data exploration) |
| **Openpyxl**               | Reading Excel files (`.xlsx` format input dataset)    |
| **Streamlit**              | Interactive web-based interface and result visualization |
| **Streamlit-option-menu**  | Sidebar navigation and menu selection in the Streamlit app |

---

## 📂 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BurakCANKURT/eeg-game-rating-classification.git
   cd eeg-game-rating-classification
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv

   # For Linux/Mac:
   source venv/bin/activate

   # For Windows:
   venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

```bash
streamlit run main.py
```

The app will open in your default browser.  
Use the **sidebar menu** to navigate between:
- 📄 Menu (project description)
- 🏁 PEGI Feature Importances
- 📊 ESRB Feature Importances
- 🤖 Outlier Detection (placeholder)

---

## 🖼️ Demo Screenshots

### 🟢 Menu Page (Project Overview):
![Menu Demo](./media/ss1.png)

---

### 🟡 PEGI Feature Importance Visualization:
![PEGI Prediction](./media/ss2.png)

---

### 🔵 ESRB Feature Importance Visualization:
![ESRB Prediction](./media/ss3.png)

---

## 📂 Project Structure

```
├── main.py                                # Streamlit app (visualization)
├── game_rating_classification.py          # Machine learning logic and model training
├── model_results                          # Each feature's importance visualization
├── requirements.txt                       # Required Python packages
├── README.md                              # Project description (this file)
└── media                                  # Program  Demo
```

---

## ⚠️ Notes

- The visualizations (`kmeans_model_*.png`) are **pre-generated** and directly displayed in the app.
- **Model training is not triggered from the Streamlit app**; only the PNG visualizations are shown.

---
##📄 Data Disclaimer:
The datasets used in this project were provided by my university professor solely for educational purposes.
These datasets are not publicly distributed and are used here for demonstration and academic showcase only.

---
## 📌 What I Learned

- Comparing multiple classification models (Random Forest, KNN, Naive Bayes) for EEG-based predictions.
- Evaluating model performance and selecting the best algorithm automatically based on accuracy.
- Understanding how EEG features contribute differently to PEGI and ESRB classifications.
- Separating model training and result visualization phases for clean project structure.
- Building an interactive dashboard with Streamlit for presenting pre-trained model outputs.
---

