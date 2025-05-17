import os

# Папка с синглами
singles_folder = "data/singles"

# Границы номеров папок
min_folder = 1
max_folder = 1747

# Список папок с более чем одним mp3 файлом
multiple_mp3_folders = []

for i in range(min_folder, max_folder + 1):
    folder_path = os.path.join(singles_folder, f"single_{i}")

    if not os.path.isdir(folder_path):
        continue  # Пропускаем, если папки нет

    # Поиск всех .mp3 файлов в папке
    mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

    if len(mp3_files) > 1:
        multiple_mp3_folders.append((folder_path, len(mp3_files)))

# Вывод списка папок с несколькими mp3 файлами
if multiple_mp3_folders:
    print("Папки с более чем одним mp3 файлом:")
    for folder, count in multiple_mp3_folders:
        print(f"{folder} - {count} mp3 файлов")
else:
    print("Во всех папках по одному или ноль mp3 файлов.")