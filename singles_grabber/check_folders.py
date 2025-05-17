import os

# Папка с синглами
singles_folder = "data/singles"

# Границы номеров папок
min_folder = 1
max_folder = 1748

# Список папок без single_ratings.json и текстового файла
missing_files = []

for i in range(min_folder, max_folder + 1):
    folder_path = os.path.join(singles_folder, f"single_{i}")

    if not os.path.isdir(folder_path):
        continue  # Пропускаем, если папки нет

    # Флаг наличия файлов
    has_json = False
    has_text = False

    # Обрабатываем файлы в папке
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".mp3"):
            new_path = os.path.join(folder_path, "track.mp3")
            os.rename(file_path, new_path)  # Переименовываем в track.mp3

        elif file == "single_ratings.json":
            has_json = True

        elif file.endswith(".txt"):
            has_text = True

    # Если нет хотя бы одного из файлов, добавляем в список
    if not has_json or not has_text:
        missing_files.append(folder_path)

# Вывод списка папок с недостающими файлами
if missing_files:
    print("Папки без single_ratings.json или .txt файла:")
    for folder in missing_files:
        print(folder)
else:
    print("Все папки содержат необходимые файлы.")