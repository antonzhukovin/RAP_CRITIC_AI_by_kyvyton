import os
import re
import lyricsgenius

# Вставьте ваш API-ключ
GENIUS_API_KEY = "ua__GajBsXr7PhgvYF3FPdvqSujalaHgKCab22PbYmSxNFN3xiA3n9Rg4UuYY2Ks"

def clean_lyrics(lyrics):
    """Очищает текст от тегов и пустых строк."""
    lines = lyrics.split("\n")
    cleaned_lyrics = [re.sub(r"\[.*?\]", "", line).strip() for line in lines]
    cleaned_lyrics = [line for line in cleaned_lyrics if line]
    return "\n".join(cleaned_lyrics)

def get_lyrics_from_genius(song_title, artist_name):
    """Ищет текст трека через Genius API по названию и исполнителю."""
    genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15, sleep_time=1)
    
    try:
        song = genius.search_song(song_title, artist_name)
        if song and song.lyrics:
            return clean_lyrics(song.lyrics)
        else:
            print("Ошибка: не удалось получить текст песни.")
            return None
    except Exception as e:
        print(f"Ошибка при получении текста с Genius: {e}")
        return None

def save_lyrics_to_folder(folder_number, lyrics):
    """Сохраняет текст трека в указанную папку."""
    folder_path = f"data/singles/single_{folder_number}"
    os.makedirs(folder_path, exist_ok=True)
    lyrics_path = os.path.join(folder_path, "text_track.txt")
    
    with open(lyrics_path, "w", encoding="utf-8") as f:
        f.write(lyrics)
    
    print(f"✅ Текст сохранён в {lyrics_path}")

def main():
    while True:
        folder_number = input("Введите номер папки (например, 1234): ").strip()
        song_title = input("Введите название трека: ").strip()
        artist_name = input("Введите имя исполнителя: ").strip()
        
        lyrics = get_lyrics_from_genius(song_title, artist_name)
        if lyrics:
            save_lyrics_to_folder(folder_number, lyrics)
        
        cont = input("Хотите добавить ещё один текст? (да/нет): ").strip().lower()
        if cont != "да":
            break

if __name__ == "__main__":
    main()