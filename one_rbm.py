import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 28*28)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))
        x = self.do(x)
        x = F.sigmoid(self.fc2(x))
        x = self.do(x)
        x = F.sigmoid(self.fc3(x))
        return x


# Загрузка данных
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root='2828', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

rbm = RBM()

# Обучение модели
optimizer = optim.Adam(rbm.parameters(), lr=0.001)
criterion = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rbm.to(device)

for epoch in range(2000):
    running_loss = 0
    for batch, labels in train_loader:
        batch, labels = batch.to(device), labels.to(device)
        batch = batch.view(-1, 28*28)
        optimizer.zero_grad()
        recon_batch = rbm.forward(batch)
        loss = criterion(recon_batch, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch {} Loss: {:.6f}'.format(epoch+1, running_loss/len(train_loader)))

# Оценка качества представления
rbm.eval()

train_dataset = datasets.ImageFolder(root='2828', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
for i in range(len(train_dataset)):
    data, _ = train_dataset[i]
    data = data.view(-1, 28*28)
    recon_data = rbm.forward(data)
    data = data.detach().numpy().reshape(28, 28)
    recon_data = recon_data.detach().numpy().reshape(28, 28)
    plt.imshow(np.concatenate((data, np.ones((28,10)), recon_data), axis=1), cmap='gray')
    plt.axis('off')
    plt.show()