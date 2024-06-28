import numpy as np
import random
import torch
import torchvision.datasets
import torch.nn.functional as F  # Добавлен импорт для функций активации
import matplotlib.pyplot as plt

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)
X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels

X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

# Переписываем устаревшие методы
X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()

# 1 вариант написания класса нейронной сети LeNet
class LeNet5_v1(torch.nn.Module):
    def __init__(self):
        super(LeNet5_v1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.Tanh()  # Заменено на Tanh
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Заменено на MaxPool2d

        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.act2 = torch.nn.Tanh()  # Заменено на Tanh
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Заменено на MaxPool2d

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = torch.nn.Tanh()  # Заменено на Tanh

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.Tanh()  # Заменено на Tanh

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

cnn_v1 = LeNet5_v1()

# 2 вариант написания класса нейронной сети LeNet
class LeNet5_v2(torch.nn.Module):
    def __init__(self):
        super(LeNet5_v2, self).__init__()
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Изменено: увеличено количество каналов
            torch.nn.ReLU(),  # Добавлен ReLU
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Изменено: увеличено количество каналов
            torch.nn.ReLU(),  # Добавлен ReLU
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.part2 = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 64, 120),  # Изменено: увеличено количество нейронов
            torch.nn.ReLU(),  # Добавлен ReLU
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),  # Добавлен ReLU
            torch.nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.part1(x)
        x = x.view(x.size(0), -1)
        x = self.part2(x)
        return x

cnn_v2 = LeNet5_v2()

# Обучение нейронной сети:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn_v1.to(device)
cnn_v2.to(device)

batch_size = 100
loss = torch.nn.CrossEntropyLoss()
optimizer_v1 = torch.optim.Adam(cnn_v1.parameters(), lr=1.0e-3)
optimizer_v2 = torch.optim.Adam(cnn_v2.parameters(), lr=1.0e-3)

test_accuracy_history_v1 = []
test_loss_history_v1 = []
test_accuracy_history_v2 = []
test_loss_history_v2 = []

X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(10):
    order = np.random.permutation(len(X_train))
    cnn_v1.train()
    cnn_v2.train()

    for start_index in range(0, len(X_train), batch_size):
        optimizer_v1.zero_grad()
        optimizer_v2.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds_v1 = cnn_v1.forward(X_batch)
        preds_v2 = cnn_v2.forward(X_batch)

        loss_value_v1 = loss(preds_v1, y_batch)
        loss_value_v2 = loss(preds_v2, y_batch)

        loss_value_v1.backward()
        loss_value_v2.backward()

        optimizer_v1.step()
        optimizer_v2.step()

    cnn_v1.eval()
    cnn_v2.eval()

    test_preds_v1 = cnn_v1.forward(X_test)
    test_loss_history_v1.append(loss(test_preds_v1, y_test).data.cpu())

    accuracy_v1 = (test_preds_v1.argmax(dim=1) == y_test).float().mean().data.cpu()
    test_accuracy_history_v1.append(accuracy_v1)

    test_preds_v2 = cnn_v2.forward(X_test)
    test_loss_history_v2.append(loss(test_preds_v2, y_test).data.cpu())

    accuracy_v2 = (test_preds_v2.argmax(dim=1) == y_test).float().mean().data.cpu()
    test_accuracy_history_v2.append(accuracy_v2)

    print(f'Epoch: {epoch}, Accuracy v1: {accuracy_v1}, Loss v1: {loss(test_preds_v1, y_test).data}, '
          f'Accuracy v2: {accuracy_v2}, Loss v2: {loss(test_preds_v2, y_test).data}')

# Тестирование на одной цифре:
Y_v1 = cnn_v1.forward(X_test[0:1, :, :, :])
Y_v2 = cnn_v2.forward(X_test[0:1, :, :, :])

np.argmax(F.softmax(Y_v1).data.cpu()[0]), np.argmax(F.softmax(Y_v2).data.cpu()[0])

plt.plot(test_accuracy_history_v1, label='v1')
plt.plot(test_accuracy_history_v2, label='v2')
plt.legend()
plt.show()

plt.plot(test_loss_history_v1, label='v1')
plt.plot(test_loss_history_v2, label='v2')
plt.legend()
plt.show()
