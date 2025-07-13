
📘 Project Documentation

**Project Title**: Credit Card Default Risk Prediction
**Author**: Omkar Hase
**Date**: July 2025

---

🔍 Problem Statement

Build a machine learning model to predict whether a credit card client will default in the next month based on demographic, financial, and payment history data.

---

 📂 Dataset

* **Source**: UCI Machine Learning Repository
* **File**: `default of credit card clients.csv`
* **Period**: April 2005 to September 2005
* **Size**: 30,000 records, 24 features
* **Target variable**: `default.payment.next.month` (0 = no default, 1 = default)

---

 📊 Features Used

| Feature Category | Example Features                                         |
| ---------------- | -------------------------------------------------------- |
| Demographic      | Sex, Education, Marriage, Age                            |
| Credit Data      | LIMIT\_BAL (Credit limit), PAY\_0 to PAY\_6 (Repayments) |
| Bill Data        | BILL\_AMT1 to BILL\_AMT6                                 |
| Payment Data     | PAY\_AMT1 to PAY\_AMT6                                   |

---

⚙️ Project Structure

```
MLendtpend/
├── app.py                      # Streamlit frontend
├── models/
│   └── model.pkl               # Trained model
├── data/
│   ├── rawdata/
│   │   └── rawdata.csv
│   └── preprocesseddata/
│       ├── x.csv
│       └── y.csv
├── src/
│   ├── datacleaning.py
│   ├── datacollection.py
│   ├── modelbuilding.py
│   └── preprocess.py
├── notebook/
│   ├── datacleaning.ipynb
│   ├── datacollection.ipynb
│   ├── eda.ipynb
│   ├── modelbuilding.ipynb
│   └── preprocess.ipynb
└── requirements.txt
```

---

🔄 Workflow

1. **Data Collection**: Loaded from UCI or local CSV.
2. **Data Cleaning**:

   * Removed nulls
   * Encoded categorical values
   * Verified data types
3. **EDA**:

   * Distribution plots
   * Correlation heatmaps
   * Default rate vs features
4. **Preprocessing**:

   * Train-test split
   * Feature scaling (e.g., `StandardScaler`)
5. **Model Building**:

   * Trained and evaluated multiple models:

     * Logistic Regression
     * Decision Tree
     * Random Forest ✅ (Best)
     * AdaBoost
     * Gradient Boosting
     * XGBoost
     * SVC
6. **Evaluation**:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 85%   |
| Precision | 86%   |
| Recall    | 83%   |
| F1-Score  | 84%   |

✅ Best model: **Random Forest**

7. **Deployment**:

   * UI built using **Streamlit**
   * Model served using `model.pkl`
   * Inputs taken from user and prediction displayed in real-time

---

 📌 Key Code Snippets

 Model Save & Load:

```python
  Save
import joblib
joblib.dump(model, "models/model.pkl")

  Load
model = joblib.load("models/model.pkl")
```

Streamlit Inputs:

```python
st.slider("Credit Limit", min_value, max_value)
st.selectbox("Sex", [1, 2])
st.selectbox("Education", [1, 2, 3, 4, 5, 6])
st.selectbox("Marriage", [1, 2, 3])
```

---

 ✅ Conclusion

* Successfully built a robust ML pipeline to predict credit card default risk.
* Deployed using Streamlit for real-time risk assessment.
* Random Forest gave the best results with 85% accuracy.

---

🔧 Future Improvements

* Use time series features (monthly trends)
* Apply SMOTE to balance data
* Tune hyperparameters with GridSearchCV
* Deploy on cloud (e.g., AWS/Heroku)
