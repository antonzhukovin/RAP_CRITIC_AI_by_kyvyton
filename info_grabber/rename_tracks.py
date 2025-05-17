import os
import glob

# Путь к каталогу с папками альбомов
BASE_PATH = "data/albums"

def rename_tracks_in_folder(folder_path):
    """
    Переименовывает все mp3 файлы в папке folder_path в формат track_XX.mp3.
    Файлы сортируются по имени.
    """
    # Получаем список всех файлов с расширением .mp3 в папке
    mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
    # Сортируем файлы (это можно изменить, если нужен другой порядок)
    mp3_files = sorted(mp3_files)
    
    for i, file_path in enumerate(mp3_files, start=1):
        new_filename = f"track_{i:02d}.mp3"
        new_path = os.path.join(folder_path, new_filename)
        try:
            os.rename(file_path, new_path)
            print(f"Переименован: {file_path} -> {new_path}")
        except Exception as e:
            print(f"Ошибка при переименовании {file_path}: {e}")

def rename_tracks_in_all_albums(base_path):
    """
    Проходит по всем папкам в base_path, начинающимся с 'album_' и переименовывает файлы в каждой папке.
    """
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.lower().startswith("album_"):
            print(f"\nОбработка папки: {folder}")
            rename_tracks_in_folder(folder_path)

if __name__ == "__main__":
    rename_tracks_in_all_albums(BASE_PATH)