# RAP_CRITIC_AI_by_kyvyton 🎤🤖

[![ЛИШИЛ ФЛОМАСТЕРА РАБОТЫ | РЗТ 4o / rztAI](https://img.youtube.com/vi/zAjkjiWMOoI/0.jpg)](https://youtu.be/zAjkjiWMOoI?si=7cnfTVcgebrJRIua "ЛИШИЛ ФЛОМАСТЕРА РАБОТЫ | РЗТ 4o / rztAI")

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
git clone https://github.com/antonzhukovin/RAP_CRITIC_AI_by_kyvyton
cd RAP_CRITIC_AI_by_kyvyton
poetry install
```
### Если Poetry не установлен:
```
pip install poetry
```

ИЛИ

```
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
