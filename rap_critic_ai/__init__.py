import os
import lyricsgenius
import joblib
from transformers import AutoTokenizer, AutoModel
from flask import Flask

# Размеры признаков
TEXT_DIM = 1024    # размер текстового эмбеддинга
AUDIO_DIM = 40     # размер MFCC-признаков

# Множитель для оценки вайба прости господи
vibe_multiplier_map = {
    1: 1.00,
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

# Пути к сохранённым скейлерам и моделям
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_SCALER_PATH = os.path.join(BASE_DIR, "assets", "scalers", "audio_scaler.pkl")
TEXT_SCALER_PATH = os.path.join(BASE_DIR, "assets", "scalers", "text_scaler.pkl")
oiewjvaoinmvaksdemvwa = "ua__GajBsXr7PhgvYF3FPdvqSujalaHgKCab22PbYmSxNFN3xiA3n9Rg4UuYY2Ks"
SINGLE_MODEL_PATH = os.path.join(BASE_DIR, "assets", "model", "music_critic_single_model.pth")
ALBUM_MODEL_PATH = os.path.join(BASE_DIR, "assets", "model", "music_critic_album_model.pth") 

genius = lyricsgenius.Genius(oiewjvaoinmvaksdemvwa, timeout=15, retries=3)
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]

# Загрузка токенизатора и модели для текстового эмбеддинга
MODEL_NAME = "ai-forever/ruRoberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text_model = AutoModel.from_pretrained(MODEL_NAME)

# Загрузка скейлеров
audio_scaler = joblib.load(AUDIO_SCALER_PATH)
text_scaler = joblib.load(TEXT_SCALER_PATH)

app = Flask(__name__)