import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from preprocessing import load_data_for_ml
from feature_extraction import extract_all_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_xgboost():
    print("XGBOOST TRAINING")

    print("\n1. Loading data")
    result = load_data_for_ml()

    if result[0] is None:
        print("Data loading failed")
        return None

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = result

    print(f"\n   Train: {len(X_train)} images")
    print(f"   Val: {len(X_val)} images")
    print(f"   Test: {len(X_test)} images")

    print("\n2. Extracting features")
    print("   Processing training images")
    X_train_feat = extract_all_features(X_train)
    print("   Processing test images")
    X_test_feat = extract_all_features(X_test)

    print(f"\n   Feature shape: {X_train_feat.shape}")

    print("\n3. Training XGBoost")
    start_time = time.time()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=1
    )

    xgb_model.fit(
        X_train_feat, y_train,
        eval_set=[(X_train_feat, y_train), (X_test_feat, y_test)],
        verbose=True
    )

    training_time = (time.time() - start_time) / 60

    print("\n4. Evaluating model...")
    y_pred = xgb_model.predict(X_test_feat)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("XGBOOST RESULTS")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f}")
    print(f"Training time: {training_time:.2f} min")

    print("\nConfusion matrix:")
    print("                Predicted")
    print("               Healthy  Blight")
    print(f"Actual Healthy:  {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"      Blight:    {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    print(f"\nTop 10 feature importances:")
    importance = xgb_model.feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"   {i + 1}. Feature {idx}: {importance[idx]:.4f}")

    os.makedirs('src/models', exist_ok=True)
    os.makedirs('src/results', exist_ok=True)

    joblib.dump(xgb_model, 'src/models/xgboost_model.pkl')
    print("\nModel saved to src/models/xgboost_model.pkl")

    with open('src/results/xgboost_results.txt', 'w', encoding='utf-8') as f:
        f.write("XGBOOST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Training time: {training_time:.2f} min\n\n")
        f.write("Confusion matrix:\n")
        f.write(f"[[{cm[0, 0]}, {cm[0, 1]}]\n")
        f.write(f" [{cm[1, 0]}, {cm[1, 1]}]]\n")

    print("Results saved to src/results/xgboost_results.txt")

    return {
        'model': 'XGBoost',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time
    }


if __name__ == "__main__":
    results = train_xgboost()