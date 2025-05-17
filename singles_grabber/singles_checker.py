import os
import json

# Папка с синглами
BASE_DIR = "data/singles"
OUTPUT_FILE = "all_tracks.txt"
missing_files = []

# Открываем файл для записи списка всех треков
with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    for i in range(1, 1748):  # Проходим по каждой папке от 1 до 1747
        folder_name = f"single_{i}"
        folder_path = os.path.join(BASE_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue  # Пропускаем, если папка не существует

        # Пути к файлам
        ratings_path = os.path.join(folder_path, "album_ratings.json")
        new_ratings_path = os.path.join(folder_path, "single_ratings.json")
        text_track_path = os.path.join(folder_path, "text_track.txt")

        # Проверяем наличие json-файла с рейтингами и файла с текстом трека
        if not os.path.exists(ratings_path) or not os.path.exists(text_track_path):
            missing_files.append(folder_name)

        # Переименовываем album_ratings.json -> single_ratings.json
        if os.path.exists(ratings_path):
            os.rename(ratings_path, new_ratings_path)

        # Читаем название и исполнителя из json-файла (если есть)
        track_name, artist_name = "Неизвестный трек", "Неизвестный исполнитель"
        if os.path.exists(new_ratings_path):
            try:
                with open(new_ratings_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    track_name = data.get("single_name", track_name)
                    artist_name = data.get("artist_name", artist_name)
            except Exception as e:
                print(f"Ошибка чтения JSON в {folder_name}: {e}")

        # Записываем в файл
        output.write(f"{i}. {artist_name} - {track_name}\n")

# Выводим список папок, где не хватает файлов
if missing_files:
    print("\nПапки с отсутствующими файлами:")
    for folder in missing_files:
        print(folder)
else:
    print("Все папки содержат необходимые файлы.")

print(f"\nСписок всех треков сохранён в {OUTPUT_FILE}.")
