import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Air Quality Prediction - Model Evaluation")


uploaded_file = st.file_uploader("E:/Air-Quality-Prediction/Data/Real-Data/Real_Combine.csv", type=["csv"])

if uploaded_file is not None:
    
    //df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    
    st.write("### Handling Missing Values")
    df['PM 2.5'] = df['PM 2.5'].fillna(df['PM 2.5'].mean())
    st.write("Missing values in 'PM 2.5' filled with column mean.")
    
    
    X = df.drop(columns=['PM 2.5'])
    y = pd.qcut(df['PM 2.5'], q=3, labels=[0, 1, 2])
    
    
    test_size = st.slider("Select test size for train-test split", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    
    model_type = st.selectbox("Select a model", ["Decision Tree", "Random Forest", "AdaBoost", "KNN"])
    
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
    elif model_type == "KNN":
        model = KNeighborsClassifier()
    
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.write(f"### Model: {model_type}")
    st.write(f"Accuracy: {accuracy:.2f}%")
    
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    
    st.write("### Make Predictions on New Data")
    user_input = {}
    for feature in X.columns:
        user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))
    
    
    user_data = pd.DataFrame([user_input])
    st.write("### User Input Data")
    st.dataframe(user_data)
    
    
    prediction = model.predict(user_data)
    
    
    prediction_meaning = {
        0: "Low PM 2.5 Level (Good Air Quality)",
        1: "Moderate PM 2.5 Level (Moderate Air Quality)",
        2: "High PM 2.5 Level (Poor Air Quality)"
    }
    
    st.write(f"### Predicted Class: {prediction[0]} ({prediction_meaning[prediction[0]]})")

else:
    st.warning("Please upload a CSV file to proceed.")
