import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# Load Dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Explore dataset
print("Dataset loaded successfully")
print("Class Distribution:\n", df['Class'].value_counts())

# Split the dataset
X = df.drop(columns=['Class'])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=10, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print(f"Classification report: \n",classification_report(y_test,y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test,rf_model.predict_proba(X_test)[:,1]):.4f}")
print("Recall (Fraud Class):",recall_score(y_test, y_pred))

# Apply smote
smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_train,y_train)

# Display new class distribution
print("\nCLass Distribution after SMOTE:\n")
print(pd.Series(y_resampled).value_counts())

# Train random forest on resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred_smote = rf_model_smote.predict(X_test)
print(f"Classification report(SMOTE): \n",classification_report(y_test,y_pred_smote))
print(f"ROC-AUC(SMOTE): {roc_auc_score(y_test,rf_model_smote.predict_proba(X_test)[:,1]):.4f}")
print("Recall (SMOTE):",recall_score(y_test, y_pred_smote))
