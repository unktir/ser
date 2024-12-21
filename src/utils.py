import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import os
import joblib
import matplotlib.pyplot as plt

# Загрузка и очистка данных
# Загрузка данных
def load_data(file_path):
    # Проверяем существование файла
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Загрузка
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data.rename(columns={"Category": "label", "Message": "message"})
    data = data[['label', 'message']]

    return data

# Очистка текста
def clean_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', text)  # Удаление ссылок
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\s+', ' ', text).strip()  # Удаление лишних пробелов
    return text

# Очистка данных
def clean_data(data):
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data['message'] = data['message'].apply(clean_text)
    return data


# Преобразование данных в нужный вид для модели
# Векторизация текста
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


# Сохранение и загрузка модели и векторизатора
# Сохранение модели и векторизатора
def save_model(model, vectorizer, model_path, vectorizer_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Загрузка модели и векторизатора
def load_model(model_path, vectorizer_path):
    # Проверяем существование файлов
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    # Загрузка
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# Отображение данных
# Отображение метрик и матрицы ошибок
def plot_metrics(y_test, predictions):
    # Метрика Accuracy для классификации
    print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')
    # Основные метрики для классификации: Precision, Recall, F1-score
    print(classification_report(y_test, predictions))

    # Матрица ошибок
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Ham", "Spam"])
    plt.title("Confusion Matrix")
    plt.show()
