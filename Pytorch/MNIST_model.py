import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchmetrics as tm

batch_size = 60
train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

def display():
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    labels

    imshow(torchvision.utils.make_grid(images))

#display()

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

       super(CNN, self).__init__()

       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
    
    def forward(self, x):
       x = F.relu(self.conv1(x))  
       x = self.pool(x)           
       x = F.relu(self.conv2(x))  
       x = self.pool(x)           
       x = x.reshape(x.shape[0], -1)  
       x = self.fc1(x)            
       return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
   print(f"Epoch [{epoch + 1}/{num_epochs}]")

   for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
       data = data.to(device)
       targets = targets.to(device)
       scores = model(data)
       loss = criterion(scores, targets)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

acc = tm.Accuracy(task="multiclass",num_classes=10)

model.eval()

with torch.no_grad():
   for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        acc(predicted,labels)


test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")

torch.save(model.state_dict(), 'MNISTmodel.pth')

