import os
import shutil
import html

TRACKS_LIST_PATH = "all_tracks.txt"
SINGLES_FOLDER = "data/singles"

def load_tracks():
    """Загружает список треков из файла, очищает от лишних пробелов и декодирует"""
    with open(TRACKS_LIST_PATH, "r", encoding="utf-8") as f:
        tracks = [html.unescape(line.strip().lower()) for line in f.readlines()]
    return tracks

def remove_duplicates(tracks):
    """Удаляет дубликаты из списка треков, но сохраняет порядок"""
    seen = set()
    unique_tracks = []
    duplicate_tracks = []

    for track in tracks:
        parts = track.split(".", 1)  # Разделение строки по первой точке
        if len(parts) != 2:
            continue  # Пропускаем строки с неправильным форматом

        track_number = parts[0].strip()  # Номер трека
        track_name = parts[1].strip()  # Полное название трека
        if track_name in seen:
            duplicate_tracks.append(track)
        else:
            seen.add(track_name)
            unique_tracks.append(track)

    return unique_tracks, duplicate_tracks

def find_matching_folder(track_number):
    """Ищет папку single_XX, соответствующую номеру трека"""
    folder_name = f"single_{track_number}"
    folder_path = os.path.join(SINGLES_FOLDER, folder_name)
    return folder_path if os.path.isdir(folder_path) else None

def delete_duplicate_folders(duplicates):
    """Удаляет папки дубликатов, если в них нет MP3-файла"""
    for track in duplicates:
        parts = track.split(".", 1)  # Разделяем по первой точке
        if len(parts) != 2:
            continue

        track_number = parts[0].strip()  # Извлекаем номер трека
        folder_path = find_matching_folder(track_number)
        
        if folder_path:
            has_mp3 = any(f.endswith(".mp3") for f in os.listdir(folder_path))
            if not has_mp3:
                shutil.rmtree(folder_path)
                print(f"Удалена папка: {folder_path}")
            else:
                print(f"Пропущена (есть MP3): {folder_path}")

def save_unique_tracks(tracks):
    """Сохраняет уникальные треки в файл"""
    with open(TRACKS_LIST_PATH, "w", encoding="utf-8") as f:
        for track in tracks:
            f.write(track + "\n")

def main():
    tracks = load_tracks()
    unique_tracks, duplicate_tracks = remove_duplicates(tracks)
    
    print(f"\nНайдено {len(duplicate_tracks)} дубликатов:")
    for d in duplicate_tracks:
        print(f"   - {d}")

    delete_duplicate_folders(duplicate_tracks)
    save_unique_tracks(unique_tracks)

    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()
