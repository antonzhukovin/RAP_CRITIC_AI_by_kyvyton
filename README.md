# RAP_CRITIC_AI_by_kyvyton 🎤🤖
https://youtu.be/zAjkjiWMOoI?si=iTiYXIuJ-a_W_WYP

AI-критик для оценки рэпа: мультимодальное приложение, анализирующее как текст, так и звук.

**by kyvyton**

---

## 🚀 Возможности

- Анализирует текст и аудиофайлы треков/альбомов
- Выдаёт оценки по 4 критериям + вайб
- Использует `ruRoberta-large` + обученную аудио-модель
- Имеет простой веб-интерфейс

---

## 📦 Установка

📌 Если у вас не установлен Python:

1. Перейдите на сайт [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Скачайте и установите последнюю версию Python для вашей операционной системы (Windows, macOS или Linux)
3. Во время установки **обязательно поставьте галочку "Add Python to PATH"**

Убедитесь, что у вас установлен Python ≥ 3.8 и [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/antonzhukovin/rapcritic.git
cd rapcritic
poetry install
```
# Если Poetry не установлен:
pip install poetry

ИЛИ

pip3 install poetry
```

---

## 🖥️ Запуск

```bash
poetry run app
```

После запуска откроется веб-интерфейс. Для остановки используйте кнопку `Завершить работу`.

---

## 📁 Структура проекта

```
rapcritic/
├── rap_critic_ai/         # Основной функционал приложения
│   ├── app.py             # Flask-приложение
│   ├── templates/         # HTML-интерфейс
│   └── ...
├── dataset/               # Подготовленные датасеты
├── model/                 # Сохранённые PyTorch модели
├── info_grabber/          # Скрипты для сбора данных
├── training/              # Код обучения модели
├── in_use/                # Временные данные (игнорируются)
├── pyproject.toml         # poetry-манифест
└── README.md
```

---

## ⚙️ Используемые технологии

- Python 3.12
- Flask
- Torch
- Transformers
- Librosa
- LyricsGenius

---

## 📄 Лицензия

MIT © kyvyton  
Если используете или модифицируете проект — указывайте автора. Благодарность в описании — приятно.