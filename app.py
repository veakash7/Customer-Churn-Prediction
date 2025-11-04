import streamlit as st
import pandas as pd
import pickle

# --- 1. LOAD THE SAVED FILES ---

# Set page config
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Load the trained model
try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'churn_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the fitted scaler
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()
    
# Load the list of processed columns
try:
    processed_columns = pickle.load(open('processed_columns.pkl', 'rb'))
except FileNotFoundError:
    st.error("Column list file 'processed_columns.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading column list: {e}")
    st.stop()

# --- 2. BUILD THE USER INTERFACE (UI) ---

st.title('Customer Churn Prediction App 🔮')
st.write("This app predicts customer churn and identifies the key risk factors based on a Logistic Regression model with 71% Recall.")
st.divider()

# --- 3. A CLEANER INPUT FORM ---
st.header('Enter Customer Details:')

with st.form("churn_form"):
    
    # Group 1: Account Info
    with st.container(border=True):
        st.subheader("Account Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider('Tenure (months)', 0, 72, 12)
            partner = st.radio('Partner', ['Yes', 'No'])
        with col2:
            monthly_charges = st.slider('Monthly Charges ($)', 0.0, 150.0, 70.0)
            dependents = st.radio('Dependents', ['Yes', 'No'])
        with col3:
            total_charges = st.slider('Total Charges ($)', 0.0, 10000.0, 1500.0)
            paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])

    # Group 2: Services
    with st.container(border=True):
        st.subheader('Service Details')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            phone_service = st.radio('Phone Service', ['Yes', 'No'])
            multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        with col2:
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            if internet_service != 'No':
                online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
                online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
            else:
                online_security = 'No internet service'
                online_backup = 'No internet service'
        with col3:
            if internet_service != 'No':
                device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
                tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
                streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
                streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
            else:
                device_protection = 'No internet service'
                tech_support = 'No internet service'
                streaming_tv = 'No internet service'
                streaming_movies = 'No internet service'

    # Group 3: Contract & Payment
    with st.container(border=True):
        st.subheader("Contract & Payment")
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        with col2:
            payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            
    # Submit button for the form
    st.divider()
    submit_button = st.form_submit_button('Predict Churn', type="primary", use_container_width=True)

# --- 4. PREPARE INPUT & DISPLAY PREDICTION ---
if submit_button:
    # --- 4a. Create Raw Data Dictionary ---
    raw_data = {
        'gender': 'Male', # Hardcoded as it wasn't a top feature
        'SeniorCitizen': 0, # Hardcoded as it wasn't a top feature
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # --- 4b. Preprocessing ---
    input_df_raw = pd.DataFrame([raw_data])
    input_df_processed = input_df_raw.copy()
    
    # One-Hot Encoding
    input_df_processed = pd.get_dummies(input_df_processed, drop_first=True)
    
    # Reindex to match the training data
    input_df_processed = input_df_processed.reindex(columns=processed_columns, fill_value=0)
    
    # Scaling
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df_processed[cols_to_scale] = scaler.transform(input_df_processed[cols_to_scale])

    # --- 4c. Make Prediction ---
    prediction = model.predict(input_df_processed)
    prediction_proba = model.predict_proba(input_df_processed)

    # --- 4d. Display Results ---
    st.divider()
    st.header("Prediction Result")
    
    col1, col2 = st.columns([1, 2]) # Make the first column smaller
    
    with col1:
        if prediction[0] == 1:
            prob = prediction_proba[0][1]
            st.metric("Churn Risk", f"{prob:.1%}", "High Risk", delta_color="inverse")
        else:
            prob = prediction_proba[0][0]
            st.metric("Loyalty Score", f"{prob:.1%}", "Low Risk", delta_color="normal")
            
    with col2:
        if prediction[0] == 1:
            st.error("**Prediction: Customer is LIKELY to Churn.**")
            st.write("**Recommendation:** This customer is at high risk. Consider offering a retention incentive, a loyalty discount, or a contract upgrade.")
        else:
            st.success("**Prediction: Customer is UNLIKELY to Churn.**")
            st.write("**Recommendation:** This customer appears loyal. Ensure continued good service.")