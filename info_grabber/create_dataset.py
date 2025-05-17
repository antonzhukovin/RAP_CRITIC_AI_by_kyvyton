import os
import glob
import json
import librosa
import numpy as np

# Путь к каталогу с папками альбомов (например, album_1, album_2, …)
BASE_PATH = "data/albums"

# Путь для сохранения датасета
OUTPUT_DATASET_PATH = "dataset"
os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

def load_album_data(album_folder):
    # 1. Объединяем все текстовые файлы track_XX.txt в один общий текст
    text_files = glob.glob(os.path.join(album_folder, "track_*.txt"))
    texts = []
    for tf in text_files:
        with open(tf, "r", encoding="utf-8") as f:
            texts.append(f.read())
    album_text = "\n".join(texts)
    
    # 2. Извлекаем аудио признаки из всех mp3-файлов
    mp3_files = glob.glob(os.path.join(album_folder, "track_*.mp3"))
    audio_features_list = []
    for mp3 in mp3_files:
        try:
            y, sr = librosa.load(mp3, sr=None)
            # Извлекаем 13 MFCC коэффициентов, усредняя по времени
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)  # вектор размерности 13
            audio_features_list.append(mfcc_mean)
        except Exception as e:
            print(f"Ошибка обработки {mp3}: {e}")
    if audio_features_list:
        album_audio_features = np.mean(np.array(audio_features_list), axis=0)
        album_audio_features = album_audio_features.tolist()  # переводим в список
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
    dataset = []
    # Ищем все папки, начинающихся с "album_"
    folders = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d)) and d.lower().startswith("album_")]
    
    total = len(folders)
    print(f"Найдено {total} папок.")
    for idx, folder in enumerate(folders, start=1):
        album_folder = os.path.join(base_path, folder)
        print(f"Обрабатывается папка {idx} из {total}: {folder}")
        album_text, album_audio_features, rating_data = load_album_data(album_folder)
        if rating_data is None:
            print(f"Пропущен альбом из папки {folder} – нет файла album_ratings.json")
            continue
        
        # Извлекаем только интересующие нас поля: criteria и vibe_multiplier
        filtered_ratings = {
            "criteria": rating_data.get("criteria", {}),
            "vibe_multiplier": rating_data.get("vibe_multiplier")
        }
        
        album_entry = {
            "album_folder": folder,
            "album_name": rating_data.get("album_name", ""),    # Для справки
            "artist_name": rating_data.get("artist_name", ""),  # Для справки
            "ratings": filtered_ratings,  # Только нужные 5 числовых значений
            "text": album_text,
            "audio_features": album_audio_features
        }
        dataset.append(album_entry)
    return dataset

if __name__ == "__main__":
    dataset = create_dataset(BASE_PATH)
    print(f"Собрано примеров: {len(dataset)}")
    
    # Сохраняем датасет в JSON-файл в папке проекта
    output_file = os.path.join(OUTPUT_DATASET_PATH, "album_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Датасет сохранён в {output_file}")