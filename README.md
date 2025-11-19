# Loan Default Prediction Model

## Business Goal
This project aims to predict whether a customer will default on a loan using classification models.  
The model is trained on customer demographic information, financial attributes, credit behavior, and loan details.  
A Flask-based REST API is provided to serve real-time predictions.

---

## Dataset Information

### **Financial Information**
- **Income** – Annual income of the customer  
- **CreditScore** – Creditworthiness score  
- **MonthsEmployed** – Employment tenure  
- **NumCreditLines** – Number of active credit lines  
- **DTIRatio** – Debt-to-income ratio  

### **Loan Information**
- **LoanID** – Unique loan identifier  
- **LoanAmount** – Amount borrowed  
- **InterestRate** – Applicable interest rate  
- **LoanTerm** – Loan duration in months  
- **HasMortgage** – Whether customer already has a mortgage (Yes/No)  
- **LoanPurpose** – Education, Car, Home, Medical, Other  
- **HasCoSigner** – Indicates presence of a co-signer (Yes/No)  

### **Output Variable (Target)**
- **Default**
  - **0 → No Default**
  - **1 → Default**

---

## Model Techniques Used
- **DictVectorizer** for feature transformation  
- **RandomForestClassifier** for classification  
- **Pickle** for model serialization  
- **Flask API** for model deployment  

---

## Training Workflow
1. Data Cleaning & Preprocessing  
2. Feature Engineering  
3. Train/Validation Split  
4. Model Training (Random Forest)  
5. Saving Model as `.bin` File  

---

## REST API Deployment
A Flask server exposes a `/predict` endpoint that accepts a JSON payload and returns the predicted probability of loan default.

**Steps:**
1. Run `train.py` from the `Script` folder to generate the model.  
2. Run `predict.py` from the `Script` folder to start the real-time Flask endpoint.  
3. Use `flask_ping.py` to validate if the Flask server is running correctly.

---

## Example JSON Payload
```
{
  "Age": 56,
  "Income": 85994,
  "LoanAmount": 50587,
  "CreditScore": 520,
  "MonthsEmployed": 80,
  "NumCreditLines": 4,
  "InterestRate": 15.23,
  "LoanTerm": 36,
  "DTIRatio": 0.44,
  "Education": "bachelor",
  "EmploymentType": "full-time",
  "MaritalStatus": "divorced",
  "HasMortgage": "yes",
  "HasDependents": "yes",
  "LoanPurpose": "other",
  "HasCoSigner": "yes"
}
```

---
## Model Testing using scripts

```
Step 1: From the Scripts folder, run train.py to generate the model.
Step 2: From the Scripts folder, run predict.py to serve the model as a real-time endpoint (RTE).
Step 3: From the Scripts folder, run model_testing.py to genrate the prediction for customer present in payload.

Note:
flask_ping.py is used to verify that the Flask API is running correctly.

```

---

## Example cURL Request
```
curl -X POST -H "Content-Type: application/json"      -d '{...}'      http://localhost:9696/predict
```

---

## Docker Instructions
```
docker build -t loan-default-model .
docker run -it -p 9696:9696 loan-default-model
```

---

## Repository Structure
```
/Notebook
/Script
    ├── train.py
    ├── predict.py
    ├── flask_ping.py
    ├── model_C=1.0.bin
README.md
```

---

## Author
Shivam Tyagi 
