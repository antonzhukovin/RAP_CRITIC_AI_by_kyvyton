import os
import json

def list_albums(base_path="data/albums"):
    # Получаем список папок, начинающихся с "album_"
    folders = [folder for folder in os.listdir(base_path) if folder.startswith("album_")]
    
    # Сортируем по числовой части имени папки
    folders.sort(key=lambda x: int(x.split("_")[1]))
    
    results = []
    for i, folder in enumerate(folders, start=1):
        json_file = os.path.join(base_path, folder, "album_ratings.json")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            artist = data.get("artist_name", "Неизвестный исполнитель")
            album = data.get("album_name", "Неизвестный альбом")
            results.append(f"{i}. {artist} - {album}")
        except Exception as e:
            print(f"Ошибка при чтении файла {json_file}: {e}")
    
    return results

if __name__ == "__main__":
    album_list = list_albums()
    output_file = "album_list.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for line in album_list:
            f.write(line + "\n")
    print(f"Список альбомов сохранён в файле: {output_file}")