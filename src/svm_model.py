import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

os.makedirs('src/models', exist_ok=True)
os.makedirs('src/results', exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def extract_hog_features(images):
    """Извлечение HOG из нормализованных изображений (0-1)"""
    features = []
    for img in images:
        # Конвертируем в uint8 (0-255) для HOG
        img_uint8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        hog_feat = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False)
        features.append(hog_feat)
    return np.array(features)


def train_svm_model():
    from preprocessing import load_and_split_data

    print("SVM MODEL TRAINING FOR TOMATO LATE BLIGHT DETECTION")

    print("\n1. Loading data")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data()

    if X_train is None:
        print("Data loading failed")
        return

    print("\n2. Extracting HOG features")
    print("   Processing training images")
    X_train_hog = extract_hog_features(X_train)
    print("   Processing test images")
    X_test_hog = extract_hog_features(X_test)

    print(f"\n   Feature shape: {X_train_hog.shape}")

    print("\n   Normalizing features with StandardScaler")
    scaler = StandardScaler()
    X_train_hog = scaler.fit_transform(X_train_hog)
    X_test_hog = scaler.transform(X_test_hog)

    print("\n3. Training SVM classifier")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_model.fit(X_train_hog, y_train)
    print("Training completed")

    print("\n4. Evaluating model")
    y_pred = svm_model.predict(X_test_hog)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nSVM MODEL RESULTS")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print("                Predicted")
    print("               Healthy  Blight")
    print(f"Actual Healthy:  {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"      Blight:    {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Blight'],
                yticklabels=['Healthy', 'Blight'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (SVM)')
    plt.tight_layout()
    plt.savefig('src/results/svm_confusion_matrix.png', dpi=150)

    joblib.dump(svm_model, 'src/models/svm_model.pkl')
    print("\nModel saved to src/models/svm_model.pkl")

    return {
        'model': 'SVM',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


if __name__ == "__main__":
    results = train_svm_model()