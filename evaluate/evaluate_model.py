import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

# Параметры (должны совпадать с параметрами обучения)
TEXT_DIM = 1024    # размер текстового эмбеддинга
AUDIO_DIM = 40     # размер MFCC-признаков

# Класс Dataset (тот же, что использовался при обучении)
class AlbumDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for album in data:
            # Пропускаем альбом, если отсутствуют текстовые или аудио признаки
            if album.get("audio_features") is None or album.get("text_embedding") is None:
                continue

            audio_feat = np.array(album["audio_features"], dtype=np.float32)
            text_feat = np.array(album["text_embedding"], dtype=np.float32)

            if len(text_feat) < TEXT_DIM:
                text_feat = np.pad(text_feat, (0, TEXT_DIM - len(text_feat)), 'constant')
            else:
                text_feat = text_feat[:TEXT_DIM]

            if len(audio_feat) < AUDIO_DIM:
                audio_feat = np.pad(audio_feat, (0, AUDIO_DIM - len(audio_feat)), 'constant')
            else:
                audio_feat = audio_feat[:AUDIO_DIM]

            criteria = album["ratings"]["criteria"]
            target = np.array([
                criteria["rhyme_imagery"],
                criteria["structure_rhythm"],
                criteria["style_execution"],
                criteria["individuality_charisma"],
                album["ratings"]["vibe_multiplier"]
            ], dtype=np.float32)
            self.samples.append((audio_feat, text_feat, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Определение модели (тот же, что использовался при обучении)
class MusicCriticModel(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=64, output_dim=5):
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
        output = self.combined(combined)
        return output

# Функция для загрузки датасета
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Функция оценки модели на тестовом наборе
def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    criterion = nn.MSELoss(reduction="sum")  # суммарная ошибка
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for audio_batch, text_batch, target_batch in dataloader:
            audio_batch = torch.tensor(audio_batch, dtype=torch.float32).to(device)
            text_batch = torch.tensor(text_batch, dtype=torch.float32).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.float32).to(device)

            outputs = model(audio_batch, text_batch)
            loss = criterion(outputs, target_batch)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Average MSE Loss on evaluation set: {avg_loss:.4f}")

    # Объединяем результаты для примера
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Выводим 5 примеров (предсказания и целевые значения)
    print("Примеры предсказаний и целевых значений:")
    for i in range(500):
        pred = np.round(all_predictions[i]).astype(int)
        target = all_targets[i].astype(int)
        print(f"Example {i+1}: Prediction: {pred}, Target: {target}")

def main():
    # Используем относительные пути
    dataset_path = os.path.join("dataset", "album_dataset_mfcc_normalized.json")
    model_path = os.path.join("model", "music_critic_model.pth")
    
    # Загрузка датасета
    dataset = load_dataset(dataset_path)
    test_dataset = AlbumDataset(dataset)
    print(f"Количество альбомов в тестовом наборе: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicCriticModel(audio_dim=AUDIO_DIM, text_dim=TEXT_DIM)
    model.to(device)
    
    # Загрузка сохраненной модели
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Модель загружена из {model_path}")

    # Оценка модели
    evaluate_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()