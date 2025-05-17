import os
import glob
import json
import numpy as np
import soundfile as sf
import openl3
from tqdm import tqdm

# Путь к каталогу с папками альбомов (например, album_1, album_2, …)
BASE_PATH = "data/albums"

# Путь для сохранения датасета
OUTPUT_DATASET_PATH = "dataset"
os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

def get_openl3_embedding(audio_path, input_repr="mel256", content_type="music", embedding_size=512):
    """
    Извлекает аудио эмбеддинг из аудиофайла с помощью OpenL3.
    
    Параметры:
      - audio_path: путь к аудиофайлу
      - input_repr: тип входного представления (например, "mel256")
      - content_type: тип контента ("music" или "speech")
      - embedding_size: размер эмбеддинга (например, 512)
    
    Возвращает усреднённый эмбеддинг (список чисел) или None в случае ошибки.
    """
    try:
        audio, sr = sf.read(audio_path)
        # Если аудио многоканальное, переводим в моно
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Извлекаем эмбеддинги OpenL3 – emb.shape будет (time_steps, embedding_size)
        emb, ts = openl3.get_audio_embedding(audio, sr, input_repr=input_repr,
                                               content_type=content_type,
                                               embedding_size=embedding_size)
        # Усредняем эмбеддинги по времени, чтобы получить один вектор
        emb_mean = np.mean(emb, axis=0)
        return emb_mean.tolist()
    except Exception as e:
        print(f"Ошибка при извлечении OpenL3 эмбеддинга из {audio_path}: {e}")
        return None

def load_album_data(album_folder):
    """
    Для одной папки альбома:
      1. Объединяет все текстовые файлы track_*.txt в один общий текст.
      2. Извлекает аудио эмбеддинг из всех mp3-файлов с помощью OpenL3 и усредняет их.
      3. Считывает оценки из файла album_ratings.json.
    Возвращает кортеж: (album_text, album_audio_features, rating_data)
    """
    # 1. Объединяем текстовые файлы
    text_files = glob.glob(os.path.join(album_folder, "track_*.txt"))
    texts = []
    for tf in text_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Ошибка чтения {tf}: {e}")
    album_text = "\n".join(texts)
    
    # 2. Извлекаем аудио эмбеддинги из mp3 файлов
    mp3_files = glob.glob(os.path.join(album_folder, "track_*.mp3"))
    audio_features_list = []
    for mp3 in mp3_files:
        features = get_openl3_embedding(mp3)
        if features is not None:
            audio_features_list.append(features)
    if audio_features_list:
        # Усредняем эмбеддинги по всем трекам альбома (по элементам)
        album_audio_features = np.mean(np.array(audio_features_list), axis=0).tolist()
    else:
        album_audio_features = None

    # 3. Считываем оценки из файла album_ratings.json
    rating_file = os.path.join(album_folder, "album_ratings.json")
    try:
        with open(rating_file, "r", encoding="utf-8") as f:
            rating_data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {rating_file}: {e}")
        rating_data = None

    return album_text, album_audio_features, rating_data

def create_dataset(base_path):
    """
    Проходит по всем папкам, начинающимся с "album_", обрабатывает данные и возвращает список словарей с информацией об альбомах.
    """
    dataset = []
    folders = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d)) and d.lower().startswith("album_")]
    
    total = len(folders)
    print(f"Найдено {total} папок.")
    for folder in tqdm(folders, desc="Обработка альбомов"):
        album_folder = os.path.join(base_path, folder)
        album_text, album_audio_features, rating_data = load_album_data(album_folder)
        if rating_data is None:
            print(f"Пропущен альбом из папки {folder} – нет файла album_ratings.json")
            continue
        
        # Извлекаем только интересующие нас поля: критерии и vibe_multiplier
        filtered_ratings = {
            "criteria": rating_data.get("criteria", {}),
            "vibe_multiplier": rating_data.get("vibe_multiplier")
        }
        
        album_entry = {
            "album_folder": folder,
            "album_name": rating_data.get("album_name", ""),
            "artist_name": rating_data.get("artist_name", ""),
            "ratings": filtered_ratings,
            "text": album_text,
            "audio_features": album_audio_features
        }
        dataset.append(album_entry)
    return dataset

if __name__ == "__main__":
    dataset = create_dataset(BASE_PATH)
    print(f"Собрано примеров: {len(dataset)}")
    
    # Сохраняем датасет в JSON-файл
    output_file = os.path.join(OUTPUT_DATASET_PATH, "album_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Датасет сохранён в {output_file}")