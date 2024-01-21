import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

LEARNING_RATE = 0.001
EPOCHS = 3000
HIDDEN_NEURONS_COUNT = 500
HIDDEN_LAYERS_COUNT = 1
DROPOUT = 0.2
IMAGES_H = 28
IMAGES_W = 28
IMAGES_DIM = IMAGES_H * IMAGES_W
class RBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.do = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(IMAGES_DIM, HIDDEN_NEURONS_COUNT)
        self.fc2 = nn.Linear(HIDDEN_NEURONS_COUNT, HIDDEN_NEURONS_COUNT)
        self.fc3 = nn.Linear(HIDDEN_NEURONS_COUNT, IMAGES_DIM)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))
        x = self.do(x)
        for i in range(HIDDEN_LAYERS_COUNT):
            x = F.sigmoid(self.fc2(x))
            x = self.do(x)
        x = F.sigmoid(self.fc3(x))
        return x

# Загрузка данных
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root='2828', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

rbm1 = RBM()
rbm2 = RBM()
rbm3 = RBM()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rbm1.to(device)
rbm2.to(device)
rbm3.to(device)

# Обучение модели 1
optimizer = optim.Adam(rbm1.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()
print("RBM 1")
for epoch in range(EPOCHS):
    running_loss = 0
    for batch, labels in train_loader:
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.view(-1, 28 * 28)
        optimizer.zero_grad()
        recon_batch = rbm1.forward(batch)
        loss = criterion(recon_batch, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('RBM 1 Epoch {} Loss: {:.6f}'.format(epoch + 1, running_loss / len(train_loader)))

# Обучение модели 2
rbm1.eval()
optimizer2 = optim.Adam(rbm2.parameters(), lr=LEARNING_RATE)
print("RBM 2")
for epoch in range(EPOCHS):
    running_loss = 0
    for batch, labels in train_loader:
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.view(-1, IMAGES_DIM)
        h1 = rbm1.forward(batch)
        optimizer2.zero_grad()
        recon_batch = rbm2.forward(h1)
        loss = criterion(recon_batch, h1)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
    print('RBM 2 Epoch {} Loss: {:.6f}'.format(epoch+1, running_loss/len(train_loader)))

# Обучение модели 3
rbm2.eval()
optimizer3 = optim.Adam(rbm3.parameters(), lr=LEARNING_RATE)
print("RBM 3")
for epoch in range(EPOCHS):
    running_loss = 0
    for batch, labels in train_loader:
        batch = batch.view(-1, IMAGES_DIM)
        h1 = rbm1.forward(batch)
        h2 = rbm2.forward(h1)
        optimizer3.zero_grad()
        recon_batch = rbm3.forward(h2)
        loss = criterion(recon_batch, h2)
        loss.backward()
        optimizer3.step()
        running_loss += loss.item()
    print('RBM 3 Epoch {} Loss: {:.6f}'.format(epoch+1, running_loss/len(train_loader)))

rbm1.eval()
rbm2.eval()
rbm3.eval()

for i in range(10):
    data, _ = train_dataset[i]
    data = data.view(-1, IMAGES_DIM)
    h1 = rbm1.forward(data)
    h2 = rbm2.forward(h1)
    recon_data = rbm3.forward(h2)
    data = data.detach().numpy().reshape(IMAGES_H, IMAGES_W)
    h1 = h1.detach().numpy().reshape(IMAGES_H, IMAGES_W)
    h2 = h2.detach().numpy().reshape(IMAGES_H, IMAGES_W)
    recon_data = recon_data.detach().numpy().reshape(IMAGES_H, IMAGES_W)
    plt.imshow(np.concatenate((data, np.ones((28,10)), h1, np.ones((28,10)), h2, np.ones((28,10)), recon_data), axis=1), cmap='gray')
    plt.axis('off')
    plt.show()



# FashionMNIST

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

for i in range(10):
    data, _ = train_dataset[i]
    data = data.view(-1, 28*28)
    recon_data = rbm2.forward(data)
    data = data.detach().numpy().reshape(28, 28)
    recon_data = recon_data.detach().numpy().reshape(28, 28)
    plt.imshow(np.concatenate((data, np.ones((28,10)), recon_data), axis=1), cmap='gray')
    plt.axis('off')
    plt.show()
