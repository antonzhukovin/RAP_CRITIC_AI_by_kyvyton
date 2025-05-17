import os
import re

BASE_PATH = "data/albums"

def rename_tracks_recursively(base_path):
    # Рекурсивно обходим все папки в BASE_PATH
    for root, dirs, files in os.walk(base_path):
        # Если имя текущей папки соответствует шаблону "album_" и содержит цифры:
        if re.match(r"album_\d+", os.path.basename(root), re.IGNORECASE):
            print(f"Обрабатываем папку: {root}")
            track_num = 1
            for filename in files:
                if filename.lower().endswith(".mp3"):
                    old_path = os.path.join(root, filename)
                    new_name = f"track_{track_num:02d}.mp3"
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(old_path, new_path)
                        print(f"Переименовано: {old_path} -> {new_path}")
                        track_num += 1
                    except Exception as e:
                        print(f"Ошибка переименования {old_path}: {e}")

if __name__ == "__main__":
    rename_tracks_recursively(BASE_PATH)