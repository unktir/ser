from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import *

# Обучение модели
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Загрузка и очистка данных
    raw_data = load_data('data/spam.csv')
    data = clean_data(raw_data)
    print(f"Total dataset size: {len(data)}")

    # Используем 80% данных для обучения, 20% для теста
    X, y = data['message'], data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Векторизация и обучение
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)

    # Визуализация метрик
    predictions = model.predict(X_test_tfidf)    
    plot_metrics(y_test, predictions)

    # Сохранение модели
    save_model(model, vectorizer, 'models/main_model/ser_model.pkl', 'models/main_model/vectorizer.pkl')
