# Customer Churn Prediction & Deployment (Python, Sklearn, Streamlit)

[Insert a GIF of your Streamlit app in action]

### 1. Project Aim & Business Value

The aim of this project is to save the company money by building a "warning system" model. This model predicts which customers are likely to churn (leave), allowing the business to proactively retain them. This is critical because acquiring new customers is far more expensive than keeping existing ones.

### 2. Data Cleaning & Preprocessing

The raw data presented two major challenges:
* **Corrupted Data:** The `TotalCharges` column was an `object` type and needed to be converted to a `float`.
* **Imbalanced Dataset:** The `Churn` variable was heavily imbalanced (5174 "No" vs 1869 "Yes"), which can ruin a model's performance.

### 3. Methodology

1.  **Handling Imbalance:** I chose **Oversampling** over Undersampling to avoid data loss. I used **SMOTE (Synthetic Minority Over-sampling Technique)**, which (like KNN) finds nearest neighbors, but acts as a "creator" to generate new, synthetic "Yes" samples.
2.  **Encoding:** I handled text data by encoding `Churn` (Yes/No -> 1/0) and using **One-Hot Encoding** (`pd.get_dummies`) for features with multiple categories.
3.  **Scaling:** All numerical features were scaled using `StandardScaler`.

### 4. Model Selection (Optimizing for Recall)

The key metric for this project is **Recall**. Why?
* A **False Negative** (our model predicts "Happy" but the customer *does* churn) is the worst-case scenario. We lose the customer forever.
* A **False Positive** (our model predicts "Churn" but the customer is happy) is a low-cost error (we give a discount to a happy customer).
* Therefore, we must use a model with **High Recall** to minimize False Negatives.

I trained and compared three models:
* **1. Logistic Regression:** **Recall = 0.71 (71%)**
* **2. Tuned Random Forest:** I used `GridSearchCV` (a brute-force optimizer) to tune this complex model, but its recall did not improve.
* **3. XGBoost:** This powerful model also failed to beat the recall of the simpler model.

**Winner:** **Logistic Regression** was the clear champion, proving that a simple, interpretable model is often the best solution.

### 5. Key Insights & "The Why"

The model's coefficients provided clear, actionable insights:
* **Top Churn Driver:** `InternetService_Fiber optic`. This is a critical problem. The company must investigate: Is the fiber service too expensive? Is it unreliable? Are competitors poaching these high-tech customers?
* **Top Retention Drivers:**
    1.  `MonthlyCharges` (negative coefficient): This indicated that customers with "bundled" services (phone + TV + internet) had higher bills but were "stuck" in the ecosystem and less likely to leave.
    2.  `tenure` & `Contract_Two year`: Long-term and 2-year contract customers are very loyal.

### 6. Deployment
The final model, scaler, and column list were saved as `.pkl` files and deployed as an interactive web app using **Streamlit**.
