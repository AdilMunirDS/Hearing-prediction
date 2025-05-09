import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

# Title
st.title("PTA Prediction from ASSR")

# Load cleaned data from file in repo
df = pd.read_excel("Final_Combined_Cleaned_No_Age_Gender.xlsx")

# Features and targets (NO ASSR_avg)
X = df[['ASSR_500Hz', 'ASSR_1KHz', 'ASSR_2KHz', 'ASSR_4KHz']]
y = df[['PTA_500Hz', 'PTA_1KHz', 'PTA_2KHz', 'PTA_4KHz']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# User input
st.header("Enter ASSR Thresholds")
assr_input = [
    st.number_input("ASSR_500Hz", 0, 120, 50),
    st.number_input("ASSR_1KHz", 0, 120, 50),
    st.number_input("ASSR_2KHz", 0, 120, 50),
    st.number_input("ASSR_4KHz", 0, 120, 50),
]
# Model selection
model_option = st.selectbox("Choose a model", ["LinearRegression", "SVM", "DecisionTree", "RandomForest", "KNN"])

# Hyperparameter tuning
params = {}
if model_option == "SVM":
    C = st.slider("C (SVM)", 0.01, 10.0, 1.0)
    kernel = st.selectbox("Kernel", ["linear", "rbf"])
    params = {'estimator__C': [C], 'estimator__kernel': [kernel]}
    model = MultiOutputRegressor(SVR())
elif model_option == "DecisionTree":
    max_depth = st.slider("Max Depth", 1, 20, 5)
    params = {'estimator__max_depth': [max_depth]}
    model = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
elif model_option == "RandomForest":
    n_estimators = st.slider("n_estimators", 10, 200, 50)
    max_depth = st.slider("Max Depth", 1, 20, 5)
    params = {'estimator__n_estimators': [n_estimators], 'estimator__max_depth': [max_depth]}
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
elif model_option == "KNN":
    n_neighbors = st.slider("n_neighbors", 1, 20, 5)
    params = {'estimator__n_neighbors': [n_neighbors]}
    model = MultiOutputRegressor(KNeighborsRegressor())
else:
    model = LinearRegression()

# Train and predict
if st.button("Train and Predict"):
    if model_option == "LinearRegression":
        model.fit(X_train, y_train)
    else:
        grid = GridSearchCV(model, param_grid=params, cv=3, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

    input_scaled = scaler.transform([assr_input])
    prediction = model.predict(input_scaled)[0]

    st.subheader("Predicted PTA Thresholds")
    freqs = ["500Hz", "1KHz", "2KHz", "4KHz"]
    for f, p in zip(freqs, prediction):
        st.write(f"PTA_{f}: {p:.2f} dB")

    # Evaluate
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    accuracy = (np.abs(y_test - y_test_pred) < 10).mean(axis=0).mean() * 100
    st.write(f"Mean Absolute Error: {mae:.2f} dB")
    st.write(f"Accuracy within ±10 dB: {accuracy:.2f}%")
