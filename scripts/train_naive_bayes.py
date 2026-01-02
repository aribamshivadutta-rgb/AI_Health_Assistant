import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import joblib
import os
from sklearn.metrics import accuracy_score

# ============================================================
# 1. SETUP PATHS (DYNAMIC & RELATIVE)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# [CRITICAL UPDATE] Point to 'chat_bot_clean'
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")

# Check if files exist
if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Training data not found in {DATA_DIR}. Please run 'scripts/chat_bot_preprocessing.py' first.")

# ============================================================
# 2. LOAD DATA
# ============================================================
print(f"Loading datasets from: {DATA_DIR}")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop("disease_id", axis=1)
y_train = train_df["disease_id"]

X_test = test_df.drop("disease_id", axis=1)
y_test = test_df["disease_id"]

# ============================================================
# 3. TRAIN NAIVE BAYES MODEL
# ============================================================
print("\nInitializing Bernoulli Naive Bayes Classifier...")
clf = BernoulliNB()
print("Training model...")
clf.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# 4. EVALUATE & SAVE
# ============================================================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Naive Bayes Accuracy on Test Set: {accuracy * 100:.2f}%")
print("-" * 30)

joblib.dump(clf, MODEL_SAVE_PATH)
print(f"\nModel saved successfully to:\n{MODEL_SAVE_PATH}")