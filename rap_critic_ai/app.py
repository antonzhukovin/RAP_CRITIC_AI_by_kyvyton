import os
import sys
import json
import re
import numpy as np
import librosa
import lyricsgenius
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, render_template, redirect, url_for
import tempfile
import math
import requests
from bs4 import BeautifulSoup
import webbrowser
import threading
import signal
import shutil
import warnings
import logging

from rap_critic_ai import (
    TEXT_DIM, 
    AUDIO_DIM,
    vibe_multiplier_map,
    BASE_DIR,
    tokenizer, 
    text_model,
    audio_scaler,
    text_scaler,
    oiewjvaoinmvaksdemvwa,
    SINGLE_MODEL_PATH,
    ALBUM_MODEL_PATH,
    genius,
    app
)

############################################
# Функции для обработки альбомов
############################################

def unzip_file(zip_path, extract_dir):
    import zipfile
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir

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
    last_empty = False
    for line in cleaned_lyrics:
        if line == "":
            if not last_empty:
                final_lyrics.append(line)
                last_empty = True
        else:
            final_lyrics.append(line)
            last_empty = False
    if final_lyrics and final_lyrics[0] == "":
        final_lyrics = final_lyrics[1:]
    return "\n".join(final_lyrics)

def save_lyrics(artist, song_title, track_number, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    song = genius.search_song(song_title, artist)
    if song:
        cleaned_text = clean_lyrics(song.lyrics)
        filename = os.path.join(output_folder, f"track_{track_number:02d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Сохранено: {filename}")
    else:
        print(f"Не удалось найти текст для {song_title}")

def save_album_lyrics(album_url, artist, output_folder, genius):
    track_names = get_album_tracks(album_url)
    if not track_names:
        print("Не удалось найти треки")
        return
    print(f"Найдено {len(track_names)} треков. Начинаю скачивание...")
    for i, track_name in enumerate(track_names, start=1):
        save_lyrics(artist, track_name, i, output_folder)

def combine_album_texts(lyrics_folder):
    texts = []
    for filename in sorted(os.listdir(lyrics_folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(lyrics_folder, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)

def process_album_audio(album_dir):
    audio_features_list = []
    for file in os.listdir(album_dir):
        if file.lower().endswith(".mp3"):
            file_path = os.path.join(album_dir, file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=AUDIO_DIM)
                mfcc_mean = np.mean(mfcc, axis=1)
                audio_features_list.append(mfcc_mean.tolist())
                print(f"{file} MFCC mean shape: {np.array(mfcc_mean).shape}")
            except Exception as e:
                print(f"Ошибка обработки {file_path}: {e}")
    if audio_features_list:
        return np.mean(np.array(audio_features_list), axis=0).tolist()
    else:
        return np.zeros(AUDIO_DIM).tolist()

def process_album(zip_dir, album_url, artist, album_name):
    lyrics_folder = os.path.join(zip_dir, "lyrics")
    save_album_lyrics(album_url, artist, lyrics_folder)
    if not os.path.exists(lyrics_folder) or not os.listdir(lyrics_folder):
        print("Папка с текстами не найдена или пуста.")
        combined_text = ""
    else:
        combined_text = combine_album_texts(lyrics_folder)
    if not combined_text.strip():
        text_embedding = np.zeros(TEXT_DIM).tolist()
    else:
        text_embedding = get_text_embedding(combined_text)
    audio_features = process_album_audio(zip_dir)
    return audio_features, text_embedding, combined_text

############################################
# Класс для обработки аудио для синглов
############################################

class AudioProcessor:
    def __init__(self, file_path, n_mfcc=40, hop_length=512):
        self.file_path = file_path
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def get_mfcc_features(self):
        try:
            y, sr = librosa.load(self.file_path, sr=None)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)
            return mfcc_mean.tolist()
        except Exception as e:
            print(f"Ошибка обработки {self.file_path}: {e}")
            return np.zeros(AUDIO_DIM).tolist()

############################################
# Класс для извлечения текстов с Genius (по URL)
############################################

class LyricsExtractor:
    def __init__(self, access_token):
        self.genius = lyricsgenius.Genius(access_token, timeout=15, retries=3)
        self.genius.skip_non_songs = True
        self.genius.excluded_terms = ["(Remix)", "(Live)"]

    def extract_lyrics_by_url(self, url: str, artist: str, title: str) -> (str, str):
        try:
            last_segment = url.strip().split("/")[-1]
            if last_segment.endswith("-lyrics"):
                base_name = last_segment[:-7]
            else:
                base_name = last_segment
            print(f"Searching for \"{title}\" by {artist} using URL...")
            song = self.genius.search_song(title, artist)
            if song:
                file_base = f"{artist}-{title}".replace(" ", "_")
                return song.lyrics, file_base
            else:
                raise ValueError("Песня не найдена на Genius.")
        except Exception as e:
            print(f"Ошибка извлечения текста из URL: {e}")
            return "", ""

############################################
# Класс для очистки текстов
############################################

class LyricsProcessor:
    def __init__(self, lyrics):
        self.lyrics = lyrics

    def clean_lyrics(self):
        lines = self.lyrics.split("\n")
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

############################################
# Функция для вычисления текстового эмбеддинга с ruRoberta-large
############################################

def get_text_embedding(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = text_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy().tolist()

############################################
# Определение модели MusicCriticModel
############################################

class MusicCriticModel(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim, output_dim=5):
        super(MusicCriticModel, self).__init__()
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, audio_input, text_input):
        audio_out = self.audio_branch(audio_input)
        text_out = self.text_branch(text_input)
        combined = torch.cat((audio_out, text_out), dim=1)
        raw_output = self.combined(combined)
        output = 1 + 9 * torch.sigmoid(raw_output)
        return output

############################################
# Flask веб-интерфейс
############################################

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return 'Выключили реп'

# Если GET-запрос содержит параметр "clear", удаляем файлы из in_use для заданной записи
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        clear_id = request.args.get('clear')
        if clear_id:
            in_use_dir = "in_use"
            if os.path.exists(in_use_dir):
                for fname in os.listdir(in_use_dir):
                    if fname.startswith(clear_id):
                        os.remove(os.path.join(in_use_dir, fname))
            return redirect(url_for('index'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if request.method == 'POST':
        record_type = request.form.get('record_type', 'single')  # 'single' или 'album'
        album_or_song_name = request.form['album_name']
        artist_name = request.form['artist_name']
        genius_url = request.form.get('genius_url', '').strip()  # Если ссылка не указана, текст не извлекается
        dataset_entry = {}
        base_name = f"{artist_name}-{album_or_song_name}".replace(" ", "_")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if record_type == 'album':
                album_file = request.files['mp3_file']
                if not album_file.filename.lower().endswith('.zip'):
                    return render_template("index.html", result="Ошибка: для альбома необходимо загрузить ZIP-файл.")
                zip_path = os.path.join(temp_dir, album_file.filename)
                album_file.save(zip_path)
                album_dir = unzip_file(zip_path, os.path.join(temp_dir, "album_tracks"))
                hidden_dim = 64  # для альбомов используем меньшую размерность скрытого слоя
                lyrics_folder = os.path.join(temp_dir, "lyrics")
                audio_features, text_embedding, combined_text = process_album(album_dir, genius_url, artist_name, album_or_song_name)
                dataset_entry = {
                    "album_name": album_or_song_name,
                    "artist_name": artist_name,
                    "text": combined_text,
                    "audio_features": audio_features,
                    "text_embedding": text_embedding
                }
                model_path = ALBUM_MODEL_PATH
            else:
                mp3_file = request.files['mp3_file']
                if not mp3_file.filename.lower().endswith('.mp3'):
                    return render_template("index.html", result="Ошибка: для сингла необходимо загрузить MP3-файл.")
                mp3_path = os.path.join(temp_dir, mp3_file.filename)
                mp3_file.save(mp3_path)
                if genius_url != "":
                    extractor = LyricsExtractor(oiewjvaoinmvaksdemvwa)
                    lyrics, _ = extractor.extract_lyrics_by_url(genius_url, artist_name, album_or_song_name)
                    if not lyrics:
                        return render_template("index.html", result="Ошибка при извлечении текста с Genius.")
                else:
                    return render_template("index.html", result="Ошибка: не задан URL для извлечения текста.")
                audio_processor = AudioProcessor(mp3_path)
                audio_features = audio_processor.get_mfcc_features()
                if audio_features is None:
                    return render_template("index.html", result="Ошибка при извлечении аудио признаков.")
                processor = LyricsProcessor(lyrics)
                cleaned_text = processor.clean_lyrics()
                text_embedding = get_text_embedding(cleaned_text)
                dataset_entry = {
                    "album_name": album_or_song_name,
                    "artist_name": artist_name,
                    "text": cleaned_text,
                    "audio_features": audio_features,
                    "text_embedding": text_embedding
                }
                model_path = SINGLE_MODEL_PATH
                hidden_dim = 256

            # Сохраняем исходную запись
            os.makedirs("in_use", exist_ok=True)
            new_file = os.path.join("in_use", f"{base_name}_dataset_entry.json")
            with open(new_file, "w", encoding="utf-8") as f:
                json.dump(dataset_entry, f, ensure_ascii=False, indent=4)
            # print(f"Запись датасета сохранена в: {new_file}")

            # Нормализация аудио признаков
            audio_feat_arr = np.array(dataset_entry["audio_features"]).reshape(1, -1)
            if audio_feat_arr.shape[1] != AUDIO_DIM:
                print(f"Предупреждение: аудио признаки имеют размер {audio_feat_arr.shape[1]}, ожидается {AUDIO_DIM}.")
            normalized_audio = audio_scaler.transform(audio_feat_arr).tolist()[0]
            dataset_entry["audio_features"] = normalized_audio

            # Нормализация текстового эмбеддинга
            text_feat_arr = np.array(dataset_entry["text_embedding"]).reshape(1, -1)
            if text_feat_arr.shape[1] < TEXT_DIM:
                text_feat_arr = np.pad(text_feat_arr, ((0,0),(0, TEXT_DIM - text_feat_arr.shape[1])), 'constant')
            else:
                text_feat_arr = text_feat_arr[:, :TEXT_DIM]
            normalized_text = text_scaler.transform(text_feat_arr).tolist()[0]
            dataset_entry["text_embedding"] = normalized_text

            output_file = os.path.join("in_use", f"{base_name}_dataset_entry_normalized.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset_entry, f, ensure_ascii=False, indent=4)
            # print(f"Нормализованная запись датасета сохранена в: {output_file}")

            # Подготовка признаков для модели
            audio_feat = np.array(dataset_entry["audio_features"], dtype=np.float32)
            text_feat = np.array(dataset_entry["text_embedding"], dtype=np.float32)
            if len(text_feat) < TEXT_DIM:
                text_feat = np.pad(text_feat, (0, TEXT_DIM - len(text_feat)), 'constant')
            else:
                text_feat = text_feat[:TEXT_DIM]
            if len(audio_feat) < AUDIO_DIM:
                audio_feat = np.pad(audio_feat, (0, AUDIO_DIM - len(audio_feat)), 'constant')
            else:
                audio_feat = audio_feat[:AUDIO_DIM]

            audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(device)
            text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(device)

            model = MusicCriticModel(audio_dim=AUDIO_DIM, text_dim=TEXT_DIM, hidden_dim=hidden_dim, output_dim=5)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            with torch.no_grad():
                prediction = model(audio_tensor, text_tensor)
            predicted_ratings = torch.round(prediction).cpu().numpy().astype(int).tolist()[0]

            ratings = {
                "rhyme_imagery": predicted_ratings[0],
                "structure_rhythm": predicted_ratings[1],
                "style_execution": predicted_ratings[2],
                "individuality_charisma": predicted_ratings[3],
                "vibe_multiplier": predicted_ratings[4]
            }

            base_sum = (ratings["rhyme_imagery"] +
                        ratings["structure_rhythm"] +
                        ratings["style_execution"] +
                        ratings["individuality_charisma"])
            overall_score = round(base_sum * 1.4 * vibe_multiplier_map.get(ratings["vibe_multiplier"], 1.0))
            if overall_score > 90:
                overall_score = 90

            result = {
                "ratings": ratings,
                "overall_score": overall_score,
                "record_type": record_type,
                "album_or_song_name": album_or_song_name,
                "artist_name": artist_name,
                "base_name": base_name  # для очистки файлов по нажатию кнопки
            }
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

def clear_in_use_folder():
    in_use_dir = os.path.join(BASE_DIR, "in_use")
    if os.path.exists(in_use_dir):
        shutil.rmtree(in_use_dir)
    os.makedirs(in_use_dir, exist_ok=True)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

def main():
    # Настройка логирования
    warnings.filterwarnings("ignore")
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)

    text_model.eval()

    clear_in_use_folder()
    if len(sys.argv) > 1 and sys.argv[1] == "--no-browser":
        app.run(debug=True)
    else:
        threading.Timer(1.5, open_browser).start()
        app.run(debug=False)