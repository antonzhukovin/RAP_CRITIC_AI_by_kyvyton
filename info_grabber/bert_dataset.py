import os
import json
from transformers import AutoTokenizer, AutoModel
import torch

# Задаем модель и токенизатор BERT
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def get_text_embedding(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy().tolist()

# Путь к исходному датасету
INPUT_DATASET_FILE = "dataset/album_dataset.json"
# Путь для сохранения обновленного датасета
OUTPUT_DATASET_FILE = "dataset/album_dataset_preprocessed.json"

with open(INPUT_DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Обрабатываем каждую запись и добавляем поле text_embedding
for album in dataset:
    text = album.get("text", "").strip()
    if text:
        album["text_embedding"] = get_text_embedding(text)
    else:
        album["text_embedding"] = None

with open(OUTPUT_DATASET_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Обновленный датасет сохранен в {OUTPUT_DATASET_FILE}")