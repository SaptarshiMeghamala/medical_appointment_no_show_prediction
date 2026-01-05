import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score


# load data
df = pd.read_csv('medical_appointment_no_show_prediction/medical_no_show_1000.csv')

# features & target
X = df[['age','days_before_appointment','previous_no_shows',
        'sms_received','alcoholism','diabetes','hypertension','gender']]
y = df['no_show']

# column groups
num_columns = ['age','days_before_appointment','previous_no_shows',
               'sms_received','alcoholism','diabetes','hypertension']
cat_columns = ['gender']


preprocessing = ColumnTransformer(
    transformers=[
        ('num' ,Pipeline([
        ('impute' , SimpleImputer(strategy= 'mean')),
        ('scaler' ,StandardScaler())
        ]),num_columns), 

        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoded' , OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]),cat_columns)

    ]
)

model_pipeline = Pipeline([
    ('preproseccing' , preprocessing),
    ('model' , LogisticRegression())
])

joblib.dump(model_pipeline,'no_shows_pipeline.pkl')


X_train , X_test , y_train , y_test = train_test_split(X, y ,test_size= 0.2 ,random_state=42)

model_pipeline.fit(X_train,y_train)
# predict = model_pipeline.predict(X_test)

proba = model_pipeline.predict_proba(X_test)[:,1]

threshold_predict = 0.5
y_predict = (proba >=threshold_predict).astype(int)

print(confusion_matrix(y_test, y_predict))
print(recall_score(y_test, y_predict))

try:
    age = float(input("Enter patient age: "))
    days_before = int(input("Days before appointment booked: "))
    prev_no_show = int(input("Previous no-shows count: "))
    sms = int(input("SMS received? (1 = Yes, 0 = No): "))
    alcohol = int(input("Alcoholism? (1 = Yes, 0 = No): "))
    diabetes = int(input("Diabetes? (1 = Yes, 0 = No): "))
    hypertension = int(input("Hypertension? (1 = Yes, 0 = No): "))
    gender = input("Gender (M / F): ")

    user_input = pd.DataFrame([{
        'age': age,
        'days_before_appointment': days_before,
        'previous_no_shows': prev_no_show,
        'sms_received': sms,
        'alcoholism': alcohol,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'gender': gender
    }])

    # probability
    prob = model_pipeline.predict_proba(user_input)[0][1]

    threshold = 0.4
    prediction = int(prob >= threshold)

    print(f"\nNo-Show Probability: {prob:.2f}")

    if prediction == 1:
        print(" High Risk: Patient may NOT show up")
    else:
        print(" Low Risk: Patient likely to show up")

except Exception as e:
    print("Error:", e)
