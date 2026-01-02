import os
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

# =======================
# CONFIGURATION
# =======================
# 1. Determine Dynamic Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # .../AI_Health_Assistant
BASE_DIR = PROJECT_ROOT

DATA_DIR = os.path.join(BASE_DIR, "data", "clean", "disease_and_symptom_clean")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_boosted():
    print("--- 1. Loading Data ---")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}. Run chat_bot_preprocessing.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("disease_id", axis=1)
    y_train = train_df["disease_id"]

    X_test = test_df.drop("disease_id", axis=1)
    y_test = test_df["disease_id"]

    print(f"Training on {len(X_train)} samples...")

    print("--- 2. Training LightGBM (Supercharged Mode) ---")

    # ⚡ KEY CHANGES FOR HIGHER ACCURACY ⚡
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(y_train.unique()),
        metric='multi_logloss',

        # 1. FIXED IMBALANCE (Crucial for rare diseases)
        class_weight='balanced',

        # 2. INCREASED POWER (Train longer and deeper)
        n_estimators=300,  # Was 100
        learning_rate=0.05,  # Slower learning = better precision
        num_leaves=50,  # Was default (31) - captures more complexity

        # 3. REGULARIZATION (Prevents memorizing, forces learning)
        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss'
    )

    print("--- 3. Evaluation ---")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 30)
    print(f"🚀 NEW ACCURACY: {acc * 100:.2f}%")
    print("=" * 30)

    # Show exactly where it's failing (Precision/Recall)
    # print(classification_report(y_test, y_pred))

    print("--- 4. Saving Model ---")
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_boosted()