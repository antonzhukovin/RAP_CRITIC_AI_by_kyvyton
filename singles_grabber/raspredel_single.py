import os
import shutil

# Параметры
singles_folder = "data/singles"
mp3_folders = [
    "data/singles/403-500",
    "data/singles/501-600",
    "data/singles/601-700",
    "data/singles/701-800",
    "data/singles/801-900",
    "data/singles/901-1000",
    "data/singles/1001-1100",
    "data/singles/1101-1200",
    "data/singles/1201-1300",
    "data/singles/1301-1400",
    "data/singles/1401-1500",
    "data/singles/1501-1600",
    "data/singles/1601-1700",
    "data/singles/1701-end"
]
singles_folder = "data/singles"

# Загружаем список треков (берем только строки с 403 по 1747)
with open(track_list_file, "r", encoding="utf-8") as file:
    tracks = [line.strip() for line in file.readlines()[402:1747]]

# Проход по каждому треку
for track in tracks:
    try:
        # Разбираем строку: "номер. исполнитель - название"
        parts = track.split(". ", 1)
        if len(parts) < 2:
            print(f"⚠ Пропущена строка: {track}")
            continue

        track_number = parts[0].strip()  # Номер трека
        remaining_part = parts[1]        # "исполнитель - название"

        # Разделяем на "исполнитель" и "название"
        if " - " not in remaining_part:
            print(f"❌ Ошибка формата: {track}")
            continue
        
        _, track_name = remaining_part.split(" - ", 1)
        track_name = track_name.strip()

        # Формируем возможные названия файлов
        filename_variants = [
            f"{track_name}.mp3",
            f"{track_name.lower()}.mp3",
            f"{track_name.replace(' ', '_')}.mp3",
            f"{track_name.replace(' ', '')}.mp3"
        ]

        # Ищем файл
        mp3_path = None
        for folder in mp3_folders:
            for variant in filename_variants:
                possible_path = os.path.join(folder, variant)
                if os.path.exists(possible_path):
                    mp3_path = possible_path
                    break
            if mp3_path:
                break

        if not mp3_path:
            print(f"❌ MP3 не найден: {track_name}")
            continue

        # Определяем папку назначения
        single_folder = os.path.join(singles_folder, f"single_{track_number}")
        if not os.path.exists(single_folder):
            print(f"❌ Папка {single_folder} не найдена, пропускаем.")
            continue

        # Перемещаем трек
        destination = os.path.join(single_folder, "track.mp3")
        shutil.move(mp3_path, destination)
        print(f"✅ Перемещено: {track_name} → {single_folder}")

        # Удаляем использованный MP3
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
            print(f"🗑 Удален: {mp3_path}")

    except Exception as e:
        print(f"Ошибка при обработке {track}: {e}")

print("🎵 Распределение завершено!")