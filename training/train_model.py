import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Наш датасет извлекает аудио-фичи, текстовые эмбеддинги и целевые оценки (целые числа от 1 до 10)
class AlbumDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.audio_features = []
        self.text_features = []
        # Целевая метка – 5 оценок: 4 критерия и vibe_multiplier (в виде целого числа от 1 до 10)
        self.targets = []
        for album in self.data:
            audio = np.array(album["audio_features"], dtype=np.float32)
            text = np.array(album["text_embedding"], dtype=np.float32)
            ratings = album["ratings"]
            target = np.array([
                ratings["criteria"]["rhyme_imagery"],
                ratings["criteria"]["structure_rhythm"],
                ratings["criteria"]["style_execution"],
                ratings["criteria"]["individuality_charisma"],
                ratings["vibe_multiplier"]
            ], dtype=np.int64)  # int64 для CrossEntropyLoss
            self.audio_features.append(audio)
            self.text_features.append(text)
            self.targets.append(target)
        
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        return self.audio_features[idx], self.text_features[idx], self.targets[idx]

# Модель с двумя ветвями (аудио и текст) и пятью классификационными головами
class MusicCriticModel(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=256, num_classes=10):
        super(MusicCriticModel, self).__init__()
        # Аудио-ветвь
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        # Текстовая ветвь
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        # Объединяем выходы обеих ветвей
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Пять классификационных голов (по одной для каждой оценки)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(5)])
        
    def forward(self, audio, text):
        audio_out = self.audio_branch(audio)
        text_out = self.text_branch(text)
        combined = torch.cat([audio_out, text_out], dim=1)
        features = self.combined(combined)
        outputs = [head(features) for head in self.heads]
        return outputs

# Гиперпараметры
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DATASET_PATH = "dataset/album_dataset_mfcc_normalized.json"
MODEL_SAVE_PATH = "music_critic_model.pth"

# Создаем датасет и DataLoader
dataset = AlbumDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

audio_dim = len(dataset.audio_features[0])
text_dim = len(dataset.text_features[0])

# Создаем модель
model = MusicCriticModel(audio_dim, text_dim, hidden_dim=256, num_classes=10)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # используется для каждой из 5 классификационных голов

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Обучающий цикл
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for audio_batch, text_batch, target_batch in dataloader:
        audio_batch = torch.tensor(audio_batch).to(device)
        text_batch = torch.tensor(text_batch).to(device)
        # CrossEntropyLoss требует классы от 0 до 9, поэтому вычтем 1
        target_batch = torch.tensor(target_batch, dtype=torch.long).to(device) - 1
        
        optimizer.zero_grad()
        outputs = model(audio_batch, text_batch)
        loss = 0.0
        for i in range(5):
            loss += criterion(outputs[i], target_batch[:, i])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * audio_batch.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss:.4f}")

print("Обучение завершено!")

# Сохраняем модель
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Модель сохранена в {MODEL_SAVE_PATH}")