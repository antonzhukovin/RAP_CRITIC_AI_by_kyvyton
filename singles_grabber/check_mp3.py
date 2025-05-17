import os

# Параметры
singles_folder = "data/singles"
max_folder = 1747
track_list_file = "all_tracks.txt"

# Загрузка списка треков из указанного файла
with open(track_list_file, "r", encoding="utf-8") as f:
    track_list = [line.strip().split(". ", 1) for line in f.readlines() if ". " in line]
track_dict = {int(num): name for num, name in track_list}

# Поиск папок без MP3
missing_mp3 = []
missing_tracks = []

for i in range(1, max_folder + 1):
    folder_path = os.path.join(singles_folder, f"single_{i}")

    if not os.path.exists(folder_path):
        print(f"⚠ Папка не найдена: {folder_path}")
        continue

    # Проверяем, есть ли в папке хоть один MP3-файл
    mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]

    if not mp3_files:
        missing_mp3.append(folder_path)
        if i in track_dict:
            missing_tracks.append(f"{i}. {track_dict[i]}")

# Вывод результата
if missing_mp3:
    print("\n❌ Папки без MP3-файла:")
    for folder in missing_mp3:
        print(folder)

if missing_tracks:
    print("\n🎵 Отсутствующие треки:")
    for track in missing_tracks:
        print(track)
else:
    print("\n✅ Все треки на месте.")

print("🔍 Проверка завершена.")