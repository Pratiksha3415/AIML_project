import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Load Data
# -------------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Save PassengerId for submission
passenger_ids = test_df["PassengerId"]

# -------------------------------
# STEP 2: Combine for preprocessing
# -------------------------------
full_data = pd.concat([train_df, test_df], sort=False)

# -------------------------------
# STEP 3: Preprocessing
# -------------------------------
full_data['Age'].fillna(full_data['Age'].median(), inplace=True)
full_data['Fare'].fillna(full_data['Fare'].median(), inplace=True)
full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
full_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical
full_data['Sex'] = full_data['Sex'].map({'male': 0, 'female': 1})
full_data = pd.get_dummies(full_data, columns=['Embarked'], drop_first=True)

# Feature Engineering
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch']

# -------------------------------
# STEP 4: Split back
# -------------------------------
train_processed = full_data[full_data['Survived'].notnull()]
test_processed = full_data[full_data['Survived'].isnull()]

X = train_processed.drop(['Survived', 'PassengerId'], axis=1)
y = train_processed['Survived'].astype(int)

X_test_final = test_processed.drop(['Survived', 'PassengerId'], axis=1)

# -------------------------------
# STEP 5: Train/Test Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 6: Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_final = scaler.transform(X_test_final)

# -------------------------------
# STEP 7: Model (HIGH ACCURACY)
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# STEP 8: Evaluation
# -------------------------------
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# STEP 9: Predict on test.csv
# -------------------------------
test_predictions = model.predict(X_test_final)

# -------------------------------
# STEP 10: Save CSV (IMPORTANT ⭐)
# -------------------------------
# STEP 10: Create FINAL CSV (Full Data + Prediction + Type)
# -------------------------------

# Copy original test data
output_df = test_df.copy()

# Add predictions
output_df["Survived"] = test_predictions

# Function to define passenger type
def get_passenger_type(row):
    gender = row['Sex'].capitalize()
    
    if row['Pclass'] == 1:
        pclass = "1st Class"
    elif row['Pclass'] == 2:
        pclass = "2nd Class"
    else:
        pclass = "3rd Class"
    
    if pd.isnull(row['Age']):
        age_group = "Unknown"
    elif row['Age'] < 18:
        age_group = "Child"
    elif row['Age'] < 50:
        age_group = "Adult"
    else:
        age_group = "Senior"
    
    return f"{gender}, {pclass}, {age_group}"

# Add Type column
output_df["Type"] = output_df.apply(get_passenger_type, axis=1)

# Reorder columns (IMPORTANT ⭐)
final_df = output_df[[
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
    "Type"
]]

# Save CSV
final_df.to_csv("survival_prediction.csv", index=False)
print("survival_prediction.csv created!")

print("\nSurvival Insights:")

# Use original training data (before scaling)
insight_df = train_processed.copy()

# Gender survival
print("Female Survival Rate:",
      round(insight_df[insight_df['Sex'] == 1]['Survived'].mean(), 2))

print("Male Survival Rate:",
      round(insight_df[insight_df['Sex'] == 0]['Survived'].mean(), 2))

# Class survival
print("1st Class Survival:",
      round(insight_df[insight_df['Pclass'] == 1]['Survived'].mean(), 2))

print("3rd Class Survival:",
      round(insight_df[insight_df['Pclass'] == 3]['Survived'].mean(), 2))

# Age-based insight
print("Average Age of Survivors:",
      round(insight_df[insight_df['Survived'] == 1]['Age'].mean(), 1))

print("Average Age of Non-Survivors:",
      round(insight_df[insight_df['Survived'] == 0]['Age'].mean(), 1))