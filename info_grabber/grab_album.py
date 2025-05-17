import math
import json
import os
import time
import re
import requests
from bs4 import BeautifulSoup
import lyricsgenius

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# ===================== PART 1: Получение оценок (risazatvorchestvo.com) =====================

def get_album_ratings_selenium(album_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Убери, чтобы видеть браузер в реальном времени
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        print("Загружаем страницу (оценки):", album_url)
        driver.get(album_url)
        time.sleep(5)  # ждём загрузки контента

        # Используем селектор для левого (серого) овала
        score_selector = (
            "body > div.lg\\:pl-\\[55px\\].flex.flex-col.h-full.min-h-screen > div > main > div > "
            "div.lg\\:p-5.lg\\:bg-zinc-900.lg\\:border.rounded-2xl.flex.max-lg\\:flex-col.relative.-mt-5.pt-5 > "
            "div.lg\\:pl-8.flex.flex-col.w-full > "
            "div.flex.max-lg\\:flex-col.items-center.lg\\:items-end.mt-auto.max-lg\\:mt-3 > "
            "div.flex.items-center.gap-3 > div:nth-child(1)"
        )
        
        try:
            score_block = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, score_selector))
            )
        except Exception as e:
            print("Не удалось найти блок с общей оценкой:", e)
            driver.quit()
            return None
        
        initial_state = score_block.get_attribute("data-state")
        print("Исходное значение data-state:", initial_state)
        
        # Получаем координаты центра блока через JS
        rect = driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            return {x: rect.left + rect.width/2, y: rect.top + rect.height/2};
        """, score_block)
        print("Координаты центра score_block:", rect)
        
        # Эмулируем события мыши через JS
        hover_script = """
            var elem = arguments[0];
            var x = arguments[1].x;
            var y = arguments[1].y;
            var ev = new MouseEvent('mousemove', { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y });
            elem.dispatchEvent(ev);
            ev = new MouseEvent('mouseenter', { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y });
            elem.dispatchEvent(ev);
            ev = new MouseEvent('mouseover', { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y });
            elem.dispatchEvent(ev);
        """
        print("Эмулируем события мыши через JavaScript...")
        driver.execute_script(hover_script, score_block, rect)
        
        # Дополнительно перемещаем мышь через ActionChains
        actions = ActionChains(driver)
        actions.move_to_element_with_offset(score_block, 5, 5).perform()
        
        # Ждем изменения data-state на "instant-open" или "delayed-open"
        print("Ждем изменения data-state на 'instant-open' или 'delayed-open'...")
        state_changed = False
        for i in range(30):  # примерно 15 секунд
            current_state = score_block.get_attribute("data-state")
            print(f"Попытка {i}: data-state = {current_state}")
            if current_state in ["instant-open", "delayed-open"]:
                state_changed = True
                break
            time.sleep(0.5)
        if not state_changed:
            print("data-state не изменился на 'instant-open' или 'delayed-open'!")
            driver.quit()
            return None
        
        # Ждем появления всплывающего блока с оценками ведущего
        print("Ждем появления блока с оценками ведущего...")
        try:
            ratings_block = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-radix-popper-content-wrapper] div.z-50"))
            )
        except Exception as e:
            print("Оценки ведущего не появились:", e)
            driver.quit()
            return None
        
        # Парсим оценки из всплывающего блока
        ratings = {}
        elements = ratings_block.find_elements(By.CSS_SELECTOR, "div.flex.justify-between.items-center.w-full.border-b")
        for elem in elements:
            try:
                category = elem.find_element(By.CSS_SELECTOR, "span.text-white.text-xs.font-semibold").text.strip()
                value = elem.find_element(By.CSS_SELECTOR, "span.font-bold").text.strip()
                ratings[category] = float(value)
            except Exception:
                continue
        
        driver.quit()
        
        if not ratings:
            print("Не удалось найти оценки!")
            return None
        
        return {
            "rhyme_imagery": ratings.get("Рифмы / образы", 0),
            "structure_rhythm": ratings.get("Структура / ритмика", 0),
            "style_execution": ratings.get("Реализация стиля", 0),
            "individuality_charisma": ratings.get("Индивидуальность / харизма", 0),
            "vibe_multiplier": ratings.get("Атмосфера / вайб", 0)
        }
    except Exception as e:
        print("Ошибка:", e)
        driver.quit()
        return None

# Пситеррор, привет :) МНОЖИТЕЛИ
VIBE_MAP = {
    1: 1.0000,
    2: 1.0675,
    3: 1.1349,
    4: 1.2024,
    5: 1.2699,
    6: 1.3373,
    7: 1.4048,
    8: 1.4723,
    9: 1.5397,
    10: 1.6072
}

def save_ratings_to_json(album_number, album_name, album_url, artist_name):
    ratings = get_album_ratings_selenium(album_url)
    if not ratings:
        return

    # Для основного файла – все показатели как int
    main_rhyme = int(ratings["rhyme_imagery"])
    main_structure = int(ratings["structure_rhythm"])
    main_style = int(ratings["style_execution"])
    main_charisma = int(ratings["individuality_charisma"])
    main_vibe = int(ratings["vibe_multiplier"])

    main_base_score = main_rhyme + main_structure + main_style + main_charisma
    main_adjusted_score = main_base_score * 1.4
    main_vibe_factor = VIBE_MAP.get(main_vibe, 1.0)
    total_score_int = math.ceil(main_adjusted_score * main_vibe_factor)

    if total_score_int > 90:
        total_score_int = 90

    json_data = {
        "album_name": album_name,
        "artist_name": artist_name,
        "criteria": {
            "rhyme_imagery": main_rhyme,
            "structure_rhythm": main_structure,
            "style_execution": main_style,
            "individuality_charisma": main_charisma
        },
        "vibe_multiplier": main_vibe,
        "total_score": total_score_int
    }

    # Для файла-бэкапа – все показатели как float
    backup_rhyme = float(ratings["rhyme_imagery"])
    backup_structure = float(ratings["structure_rhythm"])
    backup_style = float(ratings["style_execution"])
    backup_charisma = float(ratings["individuality_charisma"])
    backup_vibe = float(ratings["vibe_multiplier"])

    backup_base_score = backup_rhyme + backup_structure + backup_style + backup_charisma
    backup_adjusted_score = backup_base_score * 1.4
    backup_vibe_factor = VIBE_MAP.get(int(backup_vibe), 1.0)
    total_score_float = float(math.ceil(backup_adjusted_score * backup_vibe_factor))

    if total_score_float > 90.0:
        total_score_float = 90.0

    backup_json_data = {
        "album_name": album_name,
        "artist_name": artist_name,
        "criteria": {
            "rhyme_imagery": backup_rhyme,
            "structure_rhythm": backup_structure,
            "style_execution": backup_style,
            "individuality_charisma": backup_charisma
        },
        "vibe_multiplier": backup_vibe,
        "total_score": total_score_float
    }

    output_folder = f"data/albums/album_{album_number}"
    os.makedirs(output_folder, exist_ok=True)

    json_path = f"{output_folder}/album_ratings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"Оценки сохранены в {json_path}")

    backup_json_path = f"{output_folder}/album_ratings_backup.json"
    with open(backup_json_path, "w", encoding="utf-8") as f:
        json.dump(backup_json_data, f, indent=4, ensure_ascii=False)
    print(f"Бэкап-данные сохранены в {backup_json_path}")

    # Создаем дополнительный текстовый файл с названием альбома
    album_txt_path = f"{output_folder}/{album_name}.txt"
    with open(album_txt_path, "w", encoding="utf-8") as f:
        f.write(album_name)
    print(f"Файл с названием альбома создан: {album_txt_path}")

# ===================== PART 2: Получение текстов песен (Genius) =====================

# Genius API key
GENIUS_API_KEY = "ua__GajBsXr7PhgvYF3FPdvqSujalaHgKCab22PbYmSxNFN3xiA3n9Rg4UuYY2Ks"
genius = lyricsgenius.Genius(GENIUS_API_KEY)

def get_album_tracks(album_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(album_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        track_names = []
        for link in soup.find_all("a", class_="u-display_block"):
            track_title = link.get_text(strip=True)
            track_title = re.sub(r"\s*Lyrics\s*$", "", track_title)
            if track_title and track_title not in track_names:
                track_names.append(track_title)
        return track_names
    else:
        print("Ошибка при загрузке страницы альбома")
        return []

def clean_lyrics(lyrics):
    lines = lyrics.split("\n")
    if lines:
        lines = lines[1:]
    cleaned_lyrics = [re.sub(r"\[.*?\]", "", line).strip() for line in lines]
    final_lyrics = []
    last_line_empty = False
    for line in cleaned_lyrics:
        if line == "":
            if not last_line_empty:
                final_lyrics.append(line)
                last_line_empty = True
        else:
            final_lyrics.append(line)
            last_line_empty = False
    if final_lyrics and final_lyrics[0] == "":
        final_lyrics = final_lyrics[1:]
    return "\n".join(final_lyrics)

def save_lyrics(artist, song_title, track_number, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    song = genius.search_song(song_title, artist)
    if song:
        cleaned_text = clean_lyrics(song.lyrics)
        filename = f"{output_folder}/track_{track_number:02d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Сохранено: {filename}")
    else:
        print(f"Не удалось найти текст для {song_title}")

def save_album_lyrics(album_url, artist, output_folder):
    track_names = get_album_tracks(album_url)
    if not track_names:
        print("Не удалось найти треки")
        return
    print(f"Найдено {len(track_names)} треков. Начинаю скачивание...")
    for i, track_name in enumerate(track_names, start=1):
        save_lyrics(artist, track_name, i, output_folder)
        # time.sleep(1)

# ===================== MAIN: Объединение функционала =====================

def main():
    while True:
        album_number = input("Введите номер альбома (X): ")
        genius_album_url = input("Введите ссылку на альбом для текстов (Genius): ")
        artist_name = input("Введите имя артиста: ")
        album_name = input("Введите название альбома: ")
        ratings_url = input("Введите ссылку на альбом для оценок (risazatvorchestvo.com): ")


        # Сохраняем оценки (обе версии) и создаем файл с названием альбома
        save_ratings_to_json(album_number, album_name, ratings_url, artist_name)
        # Сохраняем тексты песен
        lyrics_output_folder = f"data/albums/album_{album_number}"
        save_album_lyrics(genius_album_url, artist_name, lyrics_output_folder)

        cont = input("Скачивание завершено. Продолжить? (y/n): ").strip().lower()
        if cont != "y":
            break

if __name__ == "__main__":
    main()
