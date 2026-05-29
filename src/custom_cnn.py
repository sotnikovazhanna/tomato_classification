import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_split_data, create_data_generators

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs('src/models', exist_ok=True)
os.makedirs('src/results', exist_ok=True)


def create_custom_cnn(input_shape=(224, 224, 3)):
    #Архитектура: 3 свёрточных блока (16→32→64) + GAP + Dense(128)
    model = models.Sequential([
        # Block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_custom_cnn():
    print("TRAINING CUSTOM CNN")

    result = load_and_split_data()
    if result[0] is None:
        print("Data loading failed")
        return None

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = result

    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size=16
    )

    model = create_custom_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint('src/models/best_custom_cnn.keras', save_best_only=True, verbose=1)
    ]

    start_time = time.time()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    X_test_norm = X_test / 255.0
    y_pred_prob = model.predict(X_test_norm)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("CUSTOM CNN RESULTS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f}")
    print(f"Train time: {train_time / 60:.2f} min")
    print(f"Parameters: {total_params:,}")

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
    plt.title('Confusion Matrix (CNN)')
    plt.tight_layout()
    plt.savefig('src/results/custom_cnn_confusion_matrix.png', dpi=150)

    model.save('src/models/custom_cnn_model.keras')

    # Training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('src/results/custom_cnn_training_history.png', dpi=150)

    with open('src/results/custom_cnn_results.txt', 'w', encoding='utf-8') as f:
        f.write("CUSTOM CNN RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Training time: {train_time / 60:.2f} min\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write("\nConfusion matrix:\n")
        f.write(f"[[{cm[0, 0]}, {cm[0, 1]}]\n")
        f.write(f" [{cm[1, 0]}, {cm[1, 1]}]]\n")

    print("Results saved to src/results/")
    print("Model saved to src/models/")

    return {
        'model': 'Custom CNN',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'parameters': total_params
    }


if __name__ == "__main__":
    train_custom_cnn()