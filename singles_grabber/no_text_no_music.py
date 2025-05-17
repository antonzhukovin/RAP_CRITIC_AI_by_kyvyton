import os

# Папка с синглами
singles_folder = "data/singles"

# Границы номеров папок
min_folder = 1
max_folder = 1748

# Список пустых папок (без .txt и .mp3)
empty_folders = []

for i in range(min_folder, max_folder + 1):
    folder_path = os.path.join(singles_folder, f"single_{i}")

    if not os.path.isdir(folder_path):
        continue  # Пропускаем, если папки нет

    # Поиск всех .txt и .mp3 файлов в папке
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

    # Если нет ни одного txt, ни одного mp3 файла, добавляем папку в список
    if not txt_files and not mp3_files:
        empty_folders.append(folder_path)

# Вывод списка пустых папок
if empty_folders:
    print("Папки без .txt и .mp3 файлов:")
    for folder in empty_folders:
        print(folder)
else:
    print("Во всех папках есть хотя бы один .txt или .mp3 файл.")