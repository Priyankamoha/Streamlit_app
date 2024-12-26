import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def load_data(file_path):
    try:
        data = pd.read_csv(r'D:\Documents\Capstone\Bank_Stability_Dataset.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        return None

def preprocess_data(data, target_column='Went_Defunct'):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Convert object columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    return model, X_train.columns

def predict_input(model, input_data, feature_columns, label_encoder):
    input_df = pd.DataFrame([input_data], columns=feature_columns).apply(pd.to_numeric, errors='coerce').fillna(0)
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

def main():
    # Custom background color styling
    st.markdown("""
    <style>
        body {
            background-color: #ffccff;
        }
        .block-container {
            background-color: #f8e1f4;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Bank Stability Prediction App")

    file_path = r'D:\Documents\Capstone\Bank_Stability_Dataset.csv'
    data = load_data(file_path)

    if data is not None:
        st.subheader("Dataset Overview")
        st.write(data.head())

        X, y_encoded, label_encoder = preprocess_data(data)
        model, feature_columns = train_model(X, y_encoded)

        st.subheader("Input Bank Details")
        with st.form("input_form"):
            input_data = {}

            # Bank Name as dropdown
            bank_names = data['Bank_Name'].unique() if 'Bank_Name' in data.columns else []
            input_data['Bank_Name'] = st.selectbox("Select Bank Name:", bank_names)

            # Collect only features up to Operational_Efficiency_Ratio
            operational_efficiency_index = feature_columns.tolist().index('Operational_Efficiency_Ratio') + 1
            selected_features = feature_columns[:operational_efficiency_index]

            for feature in selected_features:
                if feature != 'Bank_Name':
                    input_data[feature] = st.text_input(f"Enter {feature}:")

            submitted = st.form_submit_button("Predict")

            if submitted:
                try:
                    predicted_label = predict_input(model, input_data, selected_features, label_encoder)
                    st.success(f"The bank is predicted to be: {predicted_label}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
