import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Определение класса нейронной сети LeNet5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = nn.Sigmoid()

        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Sigmoid()

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x

# Создание экземпляра сети
cnn = LeNet5()

# Определение устройства (CPU или GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn.to(device)

# Параметры обучения
batch_size = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Создание загрузчиков данных
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Обучение нейронной сети
num_epochs = 10
for epoch in range(num_epochs):
    cnn.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Оценка точности без атаки
cnn.eval()
correct = 0
total = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = cnn(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy_without_attack = correct / total
print(f'Accuracy without attack: {accuracy_without_attack}')

# Функция для FGSM атаки
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed_images = fgsm_attack_helper(images, epsilon, data_grad)
    return perturbed_images

def fgsm_attack_helper(images, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

# Тестирование с атакой
epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []
adversarial_images_list = []

for epsilon in epsilon_values:
    correct = 0
    total = 0
    adversarial_images = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Создаем адверсариальные изображения с помощью FGSM
        images_adv = fgsm_attack(cnn, images, labels, epsilon)

        # Прямой проход с адверсариальными изображениями
        outputs_adv = cnn(images_adv)

        # Подсчет правильных предсказаний
        _, predicted = torch.max(outputs_adv.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Сохранение адверсариальных изображений
        adversarial_images.append(images_adv.detach().cpu())

    accuracy = correct / total
    accuracies.append(accuracy)
    adversarial_images_list.append(adversarial_images)

# Вывод результатов
plt.plot(epsilon_values, accuracies, marker='o')
plt.title('Accuracy vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.show()

# Вывод атакованных и неатакованных изображений
for i, epsilon in enumerate(epsilon_values):
    plt.figure(figsize=(15, 4))
    plt.suptitle(f'Original and Adversarial Images (Epsilon = {epsilon})')

    # Показываем первые 5 неатакованных изображений
    for j in range(5):
        plt.subplot(2, 5, j + 1)
        original_image = test_set[j][0].permute(1, 2, 0).numpy()
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

    # Показываем первые 5 атакованных изображений
    for j in range(5):
        plt.subplot(2, 5, j + 6)
        adv_image = adversarial_images_list[i][j][0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(adv_image, cmap='gray')
        plt.title('Adversarial')
        plt.axis('off')

    plt.show()
