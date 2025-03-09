import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to process data and train model
def process_data(credit_card_data):
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = LogisticRegression(max_iter=500)  # Increase max_iter to handle convergence warnings
    model.fit(X_train, Y_train)
    
    # Get accuracy
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    
    return model, training_data_accuracy, test_data_accuracy

# Function to predict fraud
def predict_fraud(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]  # Return the prediction (1 for fraud, 0 for legitimate)

# Streamlit app layout
st.title("Credit Card Fraud Detection")
st.write("A simple machine learning app to detect fraudulent transactions.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your credit card transaction dataset (CSV format)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully.")
    st.dataframe(data.head())

    # Train model
    st.write("Training the model...")
    model, train_accuracy, test_accuracy = process_data(data)
    st.success(f"Model trained successfully!")
    st.write(f"**Training Accuracy:** {train_accuracy:.2f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}")

    # Input fields for fraud detection
    st.write("Enter transaction details to check for fraud:")
    Time = st.number_input("Time", value=0.0)
    V1 = st.number_input("V1", value=0.0)
    V2 = st.number_input("V2", value=0.0)
    V3 = st.number_input("V3", value=0.0)
    V4 = st.number_input("V4", value=0.0)
    V5 = st.number_input("V5", value=0.0)
    V6 = st.number_input("V6", value=0.0)
    V7 = st.number_input("V7", value=0.0)
    V8 = st.number_input("V8", value=0.0)
    V9 = st.number_input("V9", value=0.0)
    V10 = st.number_input("V10", value=0.0)
    V11 = st.number_input("V11", value=0.0)
    V12 = st.number_input("V12", value=0.0)
    V13 = st.number_input("V13", value=0.0)
    V14 = st.number_input("V14", value=0.0)
    V15 = st.number_input("V15", value=0.0)
    V16 = st.number_input("V16", value=0.0)
    V17 = st.number_input("V17", value=0.0)
    V18 = st.number_input("V18", value=0.0)
    V19 = st.number_input("V19", value=0.0)
    V20 = st.number_input("V20", value=0.0)
    V21 = st.number_input("V21", value=0.0)
    V22 = st.number_input("V22", value=0.0)
    V23 = st.number_input("V23", value=0.0)
    V24 = st.number_input("V24", value=0.0)
    V25 = st.number_input("V25", value=0.0)
    V26 = st.number_input("V26", value=0.0)
    V27 = st.number_input("V27", value=0.0)
    V28 = st.number_input("V28", value=0.0)
    Amount = st.number_input("Amount", value=0.0)
    
    # Button to check fraud
    if st.button("Check Fraud"):
        input_data = {
            'Time': Time,
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'V7': V7,
            'V8': V8,
            'V9': V9,
            'V10': V10,
            'V11': V11,
            'V12': V12,
            'V13': V13,
            'V14': V14,
            'V15': V15,
            'V16': V16,
            'V17': V17,
            'V18': V18,
            'V19': V19,
            'V20': V20,
            'V21': V21,
            'V22': V22,
            'V23': V23,
            'V24': V24,
            'V25': V25,
            'V26': V26,
            'V27': V27,
            'V28': V28,
            'Amount': Amount
        }
        prediction = predict_fraud(model, input_data)
        if prediction == 1:
            st.error("The transaction is fraudulent.")
        else:
            st.success("The transaction is legitimate.")


