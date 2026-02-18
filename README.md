# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model
<img width="1491" height="754" alt="image" src="https://github.com/user-attachments/assets/fa731029-6c8c-4c62-a856-f952a392177f" />


## DESIGN STEPS

STEP 1: Problem Statement

Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection

Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture

Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization

Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


## Step 1: Load and Preprocess Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64*7*7)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


from torchsummary import summary

model = CNNClassifier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('Name: MOHAMMAD FAIZAL SK')
print('Register Number: 212223240092')
summary(model, input_size=(1, 28, 28))


# Initialize model, loss function, and optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: MOHAMMAD FAIZAL SK')
        print('Register Number: 212223240092')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


train_model(model, train_loader)


## Step 4: Test the Model
def test_model(model, test_loader):

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    print('Name: MOHAMMAD FAIZAL SK')
    print('Register Number: 212223240092')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    print('Name: MOHAMMAD FAIZAL SK')
    print('Register Number: 212223240092')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: MOHAMMAD FAIZAL SK')
    print('Register Number: 212223240092')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=test_dataset.classes))


test_model(model, test_loader)


## Step 5: Predict on a Single Image
def predict_image(model, image_index, dataset):

    model.eval()
    image, label = dataset[image_index]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes

    print('Name: MOHAMMAD FAIZAL SK')
    print('Register Number: 212223240092')

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')


predict_image(model, image_index=80, dataset=test_dataset)     
```

## OUTPUT
### Training Loss per Epoch

<img width="908" height="301" alt="image" src="https://github.com/user-attachments/assets/6ea55af4-4cd6-44af-95b6-fbdf29f7e48d" />




### Confusion Matrix

<img width="923" height="676" alt="image" src="https://github.com/user-attachments/assets/76bad460-db95-4ad8-9c8b-a455d1c4622b" />



### Classification Report

<img width="656" height="421" alt="image" src="https://github.com/user-attachments/assets/e068a510-d9b0-44c6-a1d0-3fea16dc3db1" />




### New Sample Data Prediction
<img width="610" height="569" alt="image" src="https://github.com/user-attachments/assets/bfabf510-f1cc-4d29-a05b-fdb5fd3459d3" />




## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
