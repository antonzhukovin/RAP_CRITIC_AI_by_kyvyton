import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

# Параметры обучения
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# Предположим, что размер текстового эмбеддинга и аудио-фичи известны
TEXT_DIM = 100   # например, длина текстового эмбеддинга
AUDIO_DIM = 40   # например, длина аудио-фичей

# Кастомный Dataset
class AlbumDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.text_features = []
        self.audio_features = []
        self.targets = []  # 5 оценок: 4 критерия + vibe_multiplier
        for album in self.data:
            text_feat = np.array(album['text_embedding'], dtype=np.float32)
            audio_feat = np.array(album['audio_features'], dtype=np.float32)
            # Можно обрезать/дополнять до TEXT_DIM и AUDIO_DIM, если необходимо
            if len(text_feat) != TEXT_DIM:
                text_feat = np.pad(text_feat, (0, max(0, TEXT_DIM - len(text_feat))), 'constant')[:TEXT_DIM]
            if len(audio_feat) != AUDIO_DIM:
                audio_feat = np.pad(audio_feat, (0, max(0, AUDIO_DIM - len(audio_feat))), 'constant')[:AUDIO_DIM]
            self.text_features.append(text_feat)
            self.audio_features.append(audio_feat)
            crit = album['ratings']['criteria']
            vibe = album['ratings']['vibe_multiplier']
            target = [
                int(crit['rhyme_imagery']) - 1,
                int(crit['structure_rhythm']) - 1,
                int(crit['style_execution']) - 1,
                int(crit['individuality_charisma']) - 1,
                int(vibe) - 1
            ]
            self.targets.append(np.array(target, dtype=np.int64))
        self.text_features = np.stack(self.text_features)
        self.audio_features = np.stack(self.audio_features)
        self.targets = np.stack(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.text_features[idx], self.audio_features[idx], self.targets[idx]

# Модель с отдельными пайплайнами для текста и аудио
class MultiModalCriticNet(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim=256):
        super(MultiModalCriticNet, self).__init__()
        # Текстовый пайплайн (можно заменить на предобученный трансформер)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Аудио пайплайн (можно заменить на CNN, предобученную на аудио)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Фьюжн (объединение представлений)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 5 "голов" для предсказания оценок (10 классов: оценки 1-10, метки 0..9)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 10) for _ in range(5)])

    def forward(self, text_feat, audio_feat):
        text_out = self.text_encoder(text_feat)
        audio_out = self.audio_encoder(audio_feat)
        # Объединяем (конкатенация вдоль последнего измерения)
        combined = torch.cat([text_out, audio_out], dim=1)
        fused = self.fusion(combined)
        # Для каждой из 5 голов получаем логиты для 10 классов
        outputs = [head(fused) for head in self.heads]
        # Результат имеет форму (batch_size, 5, 10)
        outputs = torch.stack(outputs, dim=1)
        return outputs

# Функция обучения модели
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for text_batch, audio_batch, target_batch in dataloader:
            text_batch = torch.tensor(text_batch, dtype=torch.float32).to(device)
            audio_batch = torch.tensor(audio_batch, dtype=torch.float32).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(text_batch, audio_batch)  # shape: (batch_size, 5, 10)
            loss = 0.0
            # Считаем суммарную кросс-энтропию по каждой "голове"
            for i in range(5):
                loss += criterion(outputs[:, i, :], target_batch[:, i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * text_batch.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AlbumDataset("dataset/album_dataset_mfcc_normalized.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = MultiModalCriticNet(text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, dataloader, criterion, optimizer, device, NUM_EPOCHS)

    # Сохранение модели для будущего использования
    torch.save(model.state_dict(), "multi_modal_critic_model.pth")
    print("Model saved as multi_modal_critic_model.pth")

if __name__ == "__main__":
    main()