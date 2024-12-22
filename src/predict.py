from PIL import Image
import pytesseract

# Путь к исполняемому файлу Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from utils import *

# Распознавание текста на изображении
def extract_text_from_image(image_path):
    try:
        # Открываем изображение
        image = Image.open(image_path)
        # Распознаем текст с изображения
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        return f"Ошибка: {e}"

# Предсказание
def predict_message(model, vectorizer, message):
    message = clean_text(message)
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Обработка изображений из папки и создание итогового изображения
def process_images(folder_path, model, vectorizer):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    predictions = []
    
    # Получение предсказаний для каждого изображения
    for image_path in images:
        text = extract_text_from_image(image_path)
        prediction = predict_message(model, vectorizer, text)
        print((image_path, prediction))

if __name__ == "__main__":
    # Загрузка модели
    model_version = 'main_model' # изменить при необходимости на "second_model" или на собственную
    model, vectorizer = load_model(f'models/{model_version}/ser_model.pkl', f'models/{model_version}/vectorizer.pkl')
    
    # Пользовательские сообщения в виде изображения
    process_images("images", model, vectorizer)
