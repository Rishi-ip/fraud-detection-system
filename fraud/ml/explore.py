import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

print("🔄 Step 1: Loading data...")
df = pd.read_csv('dataset/creditcard.csv')
print(f"✅ Loaded {len(df)} transactions")

print("\n🔄 Step 2: Preparing features...")
# X = everything except Class (our input)
# y = Class column (what we want to predict)
X = df.drop('Class', axis=1)
y = df['Class']
print(f"✅ Features ready — X shape: {X.shape}")

print("\n🔄 Step 3: Scaling Amount and Time...")
# V1-V28 are already scaled by the bank
# But Amount and Time are not — we fix that
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])
print("✅ Scaling done")

print("\n🔄 Step 4: Splitting into Train and Test...")
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

print("\n🔄 Step 5: Applying SMOTE to fix class imbalance...")
print(f"Before SMOTE — Fraud: {sum(y_train==1)}, Not Fraud: {sum(y_train==0)}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE  — Fraud: {sum(y_train_balanced==1)}, Not Fraud: {sum(y_train_balanced==0)}")
print("✅ SMOTE done — classes are now balanced!")

print("\n🔄 Step 6: Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_balanced, y_train_balanced)
print("✅ Model trained!")

print("\n🔄 Step 7: Evaluating model...")
y_pred = model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n🔄 Step 8: Saving model...")
os.makedirs('fraud/ml', exist_ok=True)
joblib.dump(model, 'fraud/ml/model.pkl')
print("✅ Model saved as fraud/ml/model.pkl")
print("\n🎉 ALL DONE! Your fraud detection model is ready!")