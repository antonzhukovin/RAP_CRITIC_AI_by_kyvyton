import os
from pydub import AudioSegment

def convert_all_wav_to_mp3(folder_path):
    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        # Если файл имеет расширение .wav (без учёта регистра)
        if filename.lower().endswith(".wav"):
            wav_path = os.path.join(folder_path, filename)
            # Задаем новое имя файла: заменяем .wav на .mp3
            mp3_filename = os.path.splitext(filename)[0] + ".mp3"
            mp3_path = os.path.join(folder_path, mp3_filename)
            print(f"Конвертирую: {wav_path} -> {mp3_path}")
            try:
                # Загружаем WAV и экспортируем в MP3
                audio = AudioSegment.from_wav(wav_path)
                audio.export(mp3_path, format="mp3")
            except Exception as e:
                print(f"Ошибка при конвертации {wav_path}: {e}")

if __name__ == "__main__":
    folder = input("Введите путь к папке с WAV файлами: ").strip()
    convert_all_wav_to_mp3(folder)
    print("Конвертация завершена.")