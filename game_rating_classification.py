import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import streamlit as st


class EEG:
    def __init__(self):
        self.df = pd.read_excel("04-EEG-Based Game Rating Classification (PEGI & ESRB).xlsx")     
        self.X = None
        self.y_pegi = self.df["PEGI"]
        self.y_esrb = self.df["ESRB"]
        self.model = None
        self.importances = None
        self.top_indices = None
        self.index = None
        self.X_columns = None

    def preprocess_for_pegi(self):
        
        self.df = self.Apply_KNN_imputer()
        self.X = self.df.drop(columns=["PEGI", "ESRB"])
        self.X_columns = self.X.columns
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        

    

    def preprocess_for_esrb(self):
        
        self.df = self.Apply_KNN_imputer()
        self.X = self.df.drop(columns=["PEGI", "ESRB"])
        self.X_columns = self.X.columns
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        le = LabelEncoder()
        self.y_esrb = le.fit_transform(self.y_esrb)
        
  

    def Apply_KNN_imputer(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        non_numeric_df = self.df.select_dtypes(exclude=[np.number])
        numeric_df.fillna(numeric_df.select_dtypes(include='number').mean(), inplace=True)
        best_error = float('inf')
        best_df_numeric = None

        for k in range(1,11):
            imputer = KNNImputer(n_neighbors=k)
            df_imputed = imputer.fit_transform(numeric_df)
            
            df_filled = pd.DataFrame(df_imputed, columns=numeric_df.columns)

            error = mean_squared_error(numeric_df, df_filled)

            if error < best_error:
                best_error = error
                best_df_numeric = df_filled
       
        return pd.concat([best_df_numeric, non_numeric_df], axis=1)


    def RandomForestFitTrain(self, X, y, pegi_or_esrb):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators= 100, random_state= 42)
        model.fit(X_train, y_train)
        y_pred_rf = model.predict(X_test)
        accuracy_score_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest {pegi_or_esrb} Accuracy:", accuracy_score_rf)
        print(classification_report(y_test, y_pred_rf))
        report = classification_report(y_test, y_pred_rf)
        print("KNN  Report:",report)
        return accuracy_score_rf, model


    def KNNFitTrain(self, X, y, pegi_or_esrb):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = KNeighborsClassifier(n_neighbors=5) #--> Cange it!!
        model.fit(X_train, y_train)
        y_pred_knn = model.predict(X_test)
        accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
        print(f"KNN {pegi_or_esrb} Accuracy:", accuracy_score_knn)
        report = classification_report(y_test, y_pred_knn)
        print("KNN  Report:",report)
        return accuracy_score_knn, model

    def NaiveBayesFitTrain(self, X, y, pegi_or_esrb):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred_nb = model.predict(X_test)
        accuracy_score_nb = accuracy_score(y_test, y_pred_nb)
        print(f"Naive Bayes  {pegi_or_esrb} Accuracy: ", accuracy_score_nb)
        report = classification_report(y_test, y_pred_nb)
        print("Naive Bayes Classification Report:",report)
        return accuracy_score_nb, model

    def compareReports_for_pegi(self):
        self.preprocess_for_pegi()
        random_forest_acc, rf_model = self.RandomForestFitTrain(self.X, self.y_pegi, "PEGI")
        knn_acc, knn_model = self.KNNFitTrain(self.X, self.y_pegi, "PEGI")
        naive_bayes_acc, nb_model = self.NaiveBayesFitTrain( self.X, self.y_pegi, "PEGI")

        all_model = [rf_model, knn_model, nb_model]
        all_accuracies = [random_forest_acc, knn_acc, naive_bayes_acc]
        self.best_index = all_accuracies.index(max(all_accuracies))

        self.model = all_model[self.best_index] 

    def compareReports_for_esrb(self):
        self.preprocess_for_esrb()
        random_forest_acc, rf_model = self.RandomForestFitTrain(self.X, self.y_esrb, "ESRB")
        knn_acc, knn_model = self.KNNFitTrain(self.X, self.y_esrb,"ESRB")
        naive_bayes_acc, nb_model = self.NaiveBayesFitTrain(self.X, self.y_esrb, "ESRB")

        all_model = [rf_model, knn_model, nb_model]
        all_accuracies = [random_forest_acc, knn_acc, naive_bayes_acc]
        self.best_index = all_accuracies.index(max(all_accuracies))

        self.model = all_model[self.best_index] 


    def calculateImportance_and_Visualize_for_pegi(self):
        self.compareReports_for_pegi()
        self.importances = self.model.feature_importances_
        self.top_indices = np.argsort(self.importances)[-20:]
        self.X = self.X[:, self.top_indices]

        all_models = ["RandomForest" , "KNN", "NaiveBayes"]

        if all_models[self.best_index] == "RandomForest":
            random_forest_acc, rf_model = self.RandomForestFitTrain(self.X, self.y_pegi, "PEGI")
        elif all_models[self.best_index] == "KNN":
            knn_acc, knn_model = self.KNNFitTrain(self.X, self.y_pegi, "PEGI")
        else:
            naive_bayes_acc, nb_model = self.NaiveBayesFitTrain(self.X, self.y_pegi, "PEGI")

        self.plotResult(self.importances,self.top_indices, process_name= f"{all_models[self.best_index]}_pegi")
    

    def calculateImportance_and_Visualize_for_esrb(self):
        self.compareReports_for_esrb()
        self.importances = self.model.feature_importances_
        self.top_indices = np.argsort(self.importances)[-20:]
        self.X = self.X[:, self.top_indices]
        all_models = ["RandomForest" , "KNN", "NaiveBayes"]

        if all_models[self.best_index] == "RandomForest":
            random_forest_acc, rf_model = self.RandomForestFitTrain(self.X, self.y_pegi, "ESRB")
        elif all_models[self.best_index] == "KNN":
            knn_acc, knn_model = self.KNNFitTrain(self.X, self.y_pegi, "ESRB")
        else:
            naive_bayes_acc, nb_model = self.NaiveBayesFitTrain(self.X, self.y_pegi, "ESRB")

        self.plotResult(self.importances,self.top_indices, process_name= f"{all_models[self.best_index]}_esrb")
    

    def plotResult(self,importances,top_indices, process_name):
        fig, ax = plt.subplots()
        ax.figure.set_size_inches(10, 6)  
        ax.barh(range(20), importances[top_indices])
        ax.set_yticks(range(20), [self.X_columns[i] for i in top_indices])
        ax.set_title(f"The Most Effective EEG Features in {process_name}")
        ax.set_xlabel("Feature Importance")
        st.pyplot(fig)
        fig.savefig(f"kmeans_model_{process_name}.png", bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    eeg_model = EEG()
    eeg_model.calculateImportance_and_Visualize_for_pegi()
    eeg_model.calculateImportance_and_Visualize_for_esrb()
