# medical_appointment_no_show_prediction
End-to-end machine learning project to predict medical appointment no-shows using sklearn Pipeline, ColumnTransformer, and custom probability threshold.
# Medical Appointment No-Show Prediction

This project is about predicting whether a patient will miss a medical appointment or not.

Missed appointments are a common problem in hospitals. Doctors lose time and hospitals lose money.  
The idea here is to identify patients who are more likely to not show up, so some action can be taken earlier.

---

## What this project does
Using past appointment and patient data, the model predicts:
- `0` → Patient will show up  
- `1` → Patient may not show up  

The focus is more on catching possible no-shows rather than just getting high accuracy.

---

## Features used
- Age  
- Days before appointment  
- Number of previous no-shows  
- SMS received  
- Alcoholism  
- Diabetes  
- Hypertension  
- Gender  

Target column:
- `no_show`

---

## How it works
- Logistic Regression is used for prediction  
- All preprocessing is handled using `Pipeline` and `ColumnTransformer`  
- Missing values are handled automatically  
- Numeric features are scaled  
- Categorical features are encoded  
- Instead of only using model predictions, probability (`predict_proba`) is used  

A custom probability threshold is applied to improve recall.

---

## Why recall matters here
In medical appointments, missing a no-show patient is worse than sending extra reminders.

Because of this, recall is given more importance than accuracy, and a lower threshold is used to catch more potential no-shows.

---

## How to run
1. Clone the repository  
2. Install required libraries  
3. Run the Python script  
4. Enter patient details when asked  

The model will show the probability of a no-show and give a simple risk message.

---

## What I learned
- How to build an end-to-end ML pipeline  
- How preprocessing can be automated  
- Why probability thresholds matter in real problems  
- How evaluation depends on the problem, not just accuracy  

---

## Future ideas
- Build a small web app for this project  
- Try other models and compare results  
- Improve threshold selection using graphs  
