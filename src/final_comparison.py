import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = {
    'Модель': [
        'SVM + HOG',
        'Random Forest',
        'XGBoost',
        'Custom CNN'
    ],
    'Accuracy (%)': [87.20, None, None, 98.43],
    'Precision (%)': [86.10, None, None, 100.00],
    'Recall (%)': [84.50, None, None, 97.13],
    'F1-score': [0.85, None, None, 0.9854],
    'Время обучения (мин)': [0.5, None, None, 125.15],
    'Тип модели': ['Классический', 'Ансамбль', 'Бустинг', 'Нейросетевая']
}

df = pd.DataFrame(results)

print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
print(df.to_string(index=False))

df.to_csv('results/model_comparison.csv', index=False, encoding='utf-8-sig')
print("\nТаблица сохранена в results/model_comparison.csv")