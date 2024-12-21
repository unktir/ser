# Spam Email Recognition

Проект, направленный на создание системы автоматической классификации электронных писем на категории «спам» и «не спам» с использованием методов машинного обучения и обработки естественного языка (NLP). Основная цель проекта — повышение безопасности и удобства пользователей электронной почты, а также уменьшение угроз, связанных с фишингом и другими видами нежелательных сообщений. 

## Установка (Windows)

Здесь объясняется, как создать виртуальную среду с помощью модуля python [venv](https://docs.python.org/3/tutorial/venv.html) для Windows.

1. Клонирование репозитория и переход в директорию проекта

```sh
git clone git@github.com:unktir/ser.git
cd ser
```

2. Создание и активация виртального окружения

```sh
python -m venv .venv
.venv/Scripts/activate
```

3. Обновление `pip` и установка зависимостей

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Запуск скрипта для демонстрации предсказания

```sh
python src/predict.py
```

После установки можно более подробно ознакомиться с кодом удобным для вас способом. Используя редактор кода или IDE.
