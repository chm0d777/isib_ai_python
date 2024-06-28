import numpy as np
import random
import torch
import torchvision.datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Добавляем набор данных рукописных цифр MNIST
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data.float()
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data.float()
y_test = MNIST_test.test_labels

# Преобразуем данные в размерность 6000 на 784, т.е. вытянем изображения в вектор
X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])

# Создадим класс нейронной сети
class MLP(torch.nn.Module):
    def __init__(self, input_size, n_hidden_neurons):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_hidden_neurons)
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

# Выберем "устройство" - device - на котором будут проводиться наши вычисления: CPU или GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Создадим экземпляр полносвязной нейронной сети с новыми параметрами
mlp = MLP(X_train.shape[1], 32).to(device)

# Укажем используемую функцию потерь и алгоритм оптимизации
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1.0e-4)

# Обучим нейронную сеть
batch_size = 100
n_epoch = 30

X_test = X_test.to(device)
y_test = y_test.to(device)

test_accuracy_history = []
test_loss_history = []

for epoch in range(n_epoch):
    order = np.random.permutation(len(X_train))
    
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_indexes = order[start_index:start_index + batch_size]
        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)
        preds = mlp.forward(X_batch)
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        optimizer.step()
    
    test_preds = mlp.forward(X_test)
    test_loss_history.append(loss(test_preds, y_test).data.cpu().numpy())
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
    test_accuracy_history.append(accuracy)

    print(f'Epoch: {epoch}, accuracy: {accuracy}, Loss: {loss(test_preds, y_test).data}')

# Построим графики точности и потерь
plt.plot(test_loss_history, label='Loss')
plt.plot(test_accuracy_history, label='Accuracy')
plt.legend()
plt.show()
