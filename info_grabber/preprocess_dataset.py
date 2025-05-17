import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Пути к файлам
INPUT_DATASET_FILE = "dataset/album_dataset.json"
OUTPUT_DATASET_FILE = "dataset/album_dataset_preprocessed.json"

# Задаём модель и токенизатор BERT (многоязычный вариант)
MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # переводим модель в режим инференса

def get_text_embedding(text, max_length=256):
    """
    Вычисляет эмбеддинг для текста с помощью BERT.
    Если текст длиннее max_length токенов, происходит усечение.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # Усредняем эмбеддинги по всем токенам, чтобы получить представление для всего текста
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy().tolist()

def preprocess_album(album):
    """
    Для одного альбома из датасета:
      - вычисляет эмбеддинг для объединённого текста (если текст не пустой)
      - извлекает аудио признаки
      - формирует целевой вектор из критериев и vibe_multiplier
    Возвращает словарь с предобработанными данными.
    """
    album_folder = album.get("album_folder", "")
    album_name = album.get("album_name", "")
    artist_name = album.get("artist_name", "")
    
    # Текст альбома
    album_text = album.get("text", "").strip()
    if album_text:
        text_embedding = get_text_embedding(album_text)
    else:
        text_embedding = None

    # Аудио признаки – предполагается, что это уже числовой список
    audio_features = album.get("audio_features", None)
    
    # Формируем целевой вектор из оценок: 4 критерия + vibe_multiplier
    ratings = album.get("ratings", {})
    criteria = ratings.get("criteria", {})
    vibe = ratings.get("vibe_multiplier", 0)
    target_vector = [
        criteria.get("rhyme_imagery", 0),
        criteria.get("structure_rhythm", 0),
        criteria.get("style_execution", 0),
        criteria.get("individuality_charisma", 0),
        vibe
    ]
    
    return {
        "album_folder": album_folder,
        "album_name": album_name,
        "artist_name": artist_name,
        "text_embedding": text_embedding,
        "audio_features": audio_features,
        "target_ratings": target_vector
    }

def preprocess_dataset(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    processed_data = []
    total = len(dataset)
    print(f"Обрабатывается {total} альбомов...")
    for idx, album in enumerate(dataset, start=1):
        processed_entry = preprocess_album(album)
        processed_data.append(processed_entry)
        if idx % 10 == 0:
            print(f"Обработано {idx} из {total} альбомов...")
    return processed_data

if __name__ == "__main__":
    processed_dataset = preprocess_dataset(INPUT_DATASET_FILE)
    print(f"Предобработано примеров: {len(processed_dataset)}")
    with open(OUTPUT_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_dataset, f, ensure_ascii=False, indent=4)
    print(f"Предобработанный датасет сохранён в: {OUTPUT_DATASET_FILE}")