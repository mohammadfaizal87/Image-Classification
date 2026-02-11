# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

![image](https://github.com/user-attachments/assets/2254dcdc-73bc-4bd4-8567-fe7bdc9e91bd)

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

```
class CNNClassifier(nn.Module):
    class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 7x7

        x = x.view(x.size(0), -1)                # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

```

```
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```

# Train the Model

def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('Name:MOHAMMAD FAIZAL SK')
        print('Register Number:212223240092')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
```

## OUTPUT
### Training Loss per Epoch

<img width="763" height="311" alt="Screenshot 2026-02-11 161831" src="https://github.com/user-attachments/assets/eb501c6f-dda4-4485-8437-ce46c82941a5" />



### Confusion Matrix

<img width="1227" height="685" alt="Screenshot 2026-02-11 161851" src="https://github.com/user-attachments/assets/f60baa2f-aa98-4c32-9e10-3a5e5169ab9a" />


### Classification Report

<img width="731" height="409" alt="Screenshot 2026-02-11 161904" src="https://github.com/user-attachments/assets/a8c579a5-4185-45f7-ab9e-90b8c29b55fd" />



### New Sample Data Prediction
<img width="742" height="567" alt="Screenshot 2026-02-11 161935" src="https://github.com/user-attachments/assets/bf0aafb1-f224-4c61-9c40-bc1e041d407a" />



## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
