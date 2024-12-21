from utils import *

# Предсказание
def predict_message(model, vectorizer, message):
    message = clean_text(message)
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Предсказание на пользовательских данных
def predict_messages(messages, model, vectorizer):
    for message in messages:
        prediction = predict_message(model, vectorizer, message)
        print(f"Message: {message}\nPrediction: {prediction}\n")

if __name__ == "__main__":
    # Загрузка модели
    model_version = 'main_model' # изменить при необходимости на "second_model" или на собственную
    model, vectorizer = load_model(f'models/{model_version}/ser_model.pkl', f'models/{model_version}/vectorizer.pkl')
    
    # Пользовательские сообщения
    messages = [
        "Congratulations! You've won a free ticket!",
        "Reminder: your bill is due tomorrow.",
        "Win $1000 cash now!",
        "Let's meet tomorrow at 5pm.",
    ]
    

    predict_messages(messages, model, vectorizer)
