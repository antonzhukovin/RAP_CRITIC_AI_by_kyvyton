import math
import json
import os
import time
import re
from bs4 import BeautifulSoup
import lyricsgenius

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# ===================== Глобальные переменные =====================

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

BASE_URL = "https://risazatvorchestvo.com"

# ===================== Функция для извлечения ссылок на треки из страницы рейтинга =====================

def get_track_links_from_rating(page_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # запуск в фоновом режиме
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        print("Загружаем страницу рейтинга:", page_url)
        driver.get(page_url)
        # Ждем появления заголовка "Треки"
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//h2[contains(text(), 'Треки')]"))
        )
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        # Используем аргумент string вместо text (DeprecationWarning)
        tracks_header = soup.find("h2", string="Треки")
        if not tracks_header:
            print("Не удалось найти секцию 'Треки'")
            driver.quit()
            return []
        container = tracks_header.find_parent("div", class_="flex flex-col gap-2")
        if not container:
            container = tracks_header.parent
        track_a_tags = container.find_all("a", href=re.compile(r"^/track/"))
        links = []
        for a in track_a_tags:
            href = a.get("href")
            if href and href.startswith("/track/"):
                full_url = BASE_URL + href
                if full_url not in links:
                    links.append(full_url)
        driver.quit()
        return links
    except Exception as e:
        print(f"Ошибка при загрузке страницы рейтинга: {e}")
        driver.quit()
        return []

# ===================== Функция для извлечения данных сингла (название, имя исполнителя, оценки) =====================

def fetch_single_data(album_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # фоновый режим
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        print("Загружаем страницу:", album_url)
        driver.get(album_url)
        time.sleep(5)  # Ждем загрузки динамического контента

        # Извлекаем название сингла
        try:
            album_name_elem = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/main/div/div[1]/div[3]/div[2]"))
            )
            album_name = album_name_elem.text.strip()
        except Exception as e:
            print("Не удалось извлечь название сингла по первому XPath:", e)
            album_name = None
        print("Извлечённое название сингла:", album_name)

        # Извлекаем имя исполнителя
        try:
            artist_elem = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/main/div/div[1]/div[3]/div[3]/div[1]/a/span"))
            )
            artist_name = artist_elem.text.strip()
        except Exception as e:
            print("Не удалось извлечь имя исполнителя по первому XPath:", e)
            try:
                artist_elem = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/main/div/div[1]/div[3]/div[3]/div[1]/div/span"))
                )
                artist_name = artist_elem.text.strip()
            except Exception as e:
                print("Не удалось извлечь имя исполнителя по альтернативному XPath:", e)
                artist_name = None
        print("Извлечённое имя исполнителя:", artist_name)
        
        # Извлекаем оценки
        score_selector = (
            "body > div.lg\\:pl-\\[55px\\].flex.flex-col.h-full.min-h-screen > div > main > div > "
            "div.lg\\:p-5.lg\\:bg-zinc-900.lg\\:border.rounded-2xl.flex.max-lg\\:flex-col.relative.-mt-5.pt-5 > "
            "div.lg\\:pl-8.flex.flex-col.w-full > "
            "div.flex.max-lg\\:flex-col.items-center.lg\\:items-end.mt-auto.max-lg\\:mt-3 > "
            "div.flex.items-center.gap-3 > div:nth-child(1)"
        )
        score_block = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, score_selector))
        )
        initial_state = score_block.get_attribute("data-state")
        print("Исходное значение data-state:", initial_state)
        
        rect = driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            return {x: rect.left + rect.width/2, y: rect.top + rect.height/2};
        """, score_block)
        print("Координаты центра score_block:", rect)
        
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
        
        actions = ActionChains(driver)
        actions.move_to_element_with_offset(score_block, 5, 5).perform()
        
        print("Ждем изменения data-state на 'instant-open' или 'delayed-open'...")
        state_changed = False
        for i in range(30):
            current_state = score_block.get_attribute("data-state")
            print(f"Попытка {i}: data-state = {current_state}")
            if current_state in ["instant-open", "delayed-open"]:
                state_changed = True
                break
            time.sleep(0.5)
        if not state_changed:
            print("data-state не изменился!")
            driver.quit()
            return album_name, artist_name, None
        
        print("Ждем появления блока с оценками ведущего...")
        ratings_block = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-radix-popper-content-wrapper] div.z-50"))
        )
        
        ratings = {}
        elements = ratings_block.find_elements(By.CSS_SELECTOR, "div.flex.justify-between.items-center.w-full.border-b")
        for elem in elements:
            try:
                category = elem.find_element(By.CSS_SELECTOR, "span.text-white.text-xs.font-semibold").text.strip()
                value = elem.find_element(By.CSS_SELECTOR, "span.font-bold").text.strip()
                ratings[category] = float(value)
            except Exception:
                continue
        
        if not ratings:
            print("Не удалось найти оценки!")
            driver.quit()
            return album_name, artist_name, None
        
        driver.quit()
        
        final_ratings = {
            "rhyme_imagery": ratings.get("Рифмы / образы", 0),
            "structure_rhythm": ratings.get("Структура / ритмика", 0),
            "style_execution": ratings.get("Реализация стиля", 0),
            "individuality_charisma": ratings.get("Индивидуальность / харизма", 0),
            "vibe_multiplier": ratings.get("Атмосфера / вайб", 0)
        }
        
        return album_name, artist_name, final_ratings

    except Exception as e:
        print("Ошибка:", e)
        driver.quit()
        return None, None, None

# ===================== Функция очистки текста трека =====================

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

# ===================== Функция сохранения оценок в JSON =====================

def save_ratings_to_json(single_number, single_name, ratings, artist_name):
    if ratings is None:
        print("Оценки не получены, не сохраняем JSON.")
        return

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
        "single_name": single_name,
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
        "single_name": single_name,
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

    output_folder = f"data/singles/single_{single_number}"
    os.makedirs(output_folder, exist_ok=True)

    json_path = f"{output_folder}/album_ratings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"Оценки сохранены в {json_path}")

    backup_json_path = f"{output_folder}/album_ratings_backup.json"
    with open(backup_json_path, "w", encoding="utf-8") as f:
        json.dump(backup_json_data, f, indent=4, ensure_ascii=False)
    print(f"Бэкап-данные сохранены в {backup_json_path}")

# ===================== Функция получения текста трека через Genius API =====================

def get_track_lyrics(artist, track_title):
    GENIUS_API_KEY = "ua__GajBsXr7PhgvYF3FPdvqSujalaHgKCab22PbYmSxNFN3xiA3n9Rg4UuYY2Ks"
    genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15, sleep_time=1)
    try:
        print(f'Searching for "{track_title}" by {artist}...')
        song = genius.search_song(track_title, artist)
        if song:
            print("Done.")
            return clean_lyrics(song.lyrics)
        else:
            print(f"Не удалось найти текст для трека '{track_title}' исполнителя '{artist}'")
            return None
    except Exception as e:
        print(f"Ошибка при поиске текста для трека '{track_title}' исполнителя '{artist}': {e}")
        return None

# ===================== MAIN =====================

def main():
    # Задаем диапазон страниц рейтингов:
    # Начало: 2023-10, окончание: 2021-04
    current_year = 2023
    current_month = 10
    end_year = 2021
    end_month = 7
    folder_index = 1574  # Начинаем с папки 712

    while (current_year, current_month) >= (end_year, end_month):
        # page_url = f"{BASE_URL}/rating?year={current_year}&month={current_month}"
        page_url = f"{BASE_URL}/freshmen?month={current_month}&year={current_year}"
        print("\n------------------------------------------------------")
        print("Обрабатываем страницу рейтинга:", page_url)
        track_links = get_track_links_from_rating(page_url)
        if not track_links:
            print("Не найдено треков на данной странице.")
        else:
            print(f"Найдено {len(track_links)} треков:")
            for i, link in enumerate(track_links, start=1):
                print(f"{i}: {link}")
            # Обработка каждого трека на странице
            for track_url in track_links:
                print(f"\nОбработка трека {folder_index}: {track_url}")
                album_name, artist_name, ratings = fetch_single_data(track_url)
                if not album_name:
                    album_name = input("Не удалось извлечь название сингла. Введите название сингла: ")
                if not artist_name:
                    artist_name = input("Не удалось извлечь имя исполнителя. Введите имя исполнителя: ")
                output_folder = f"data/singles/single_{folder_index}"
                os.makedirs(output_folder, exist_ok=True)

                save_ratings_to_json(folder_index, album_name, ratings, artist_name)

                if ratings:
                    print("Полученные оценки:", ratings)
                else:
                    print("Оценки не получены.")

                lyrics = get_track_lyrics(artist_name, album_name)
                if lyrics:
                    lyrics_path = f"{output_folder}/text_track.txt"
                    with open(lyrics_path, "w", encoding="utf-8") as f:
                        f.write(lyrics)
                    print(f"Текст трека сохранён в {lyrics_path}")
                else:
                    print("Текст трека не найден.")

                folder_index += 1
        # Переходим к предыдущему месяцу
        if current_month == 1:
            current_year -= 1
            current_month = 12
        else:
            current_month -= 1

if __name__ == "__main__":
    main()