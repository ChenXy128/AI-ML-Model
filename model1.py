from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import resample
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torchvision import transforms

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CNN(nn.Module):
    
    def __init__(self, classes, drop_prob):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.bn3 = nn.BatchNorm1d(64)  
        self.fc2 = nn.Linear(64, classes)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.bn3(self.fc1(x))
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
    

class DataLoaderHeler(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

class Model:  
    """
    This class represents an AI model.
    """
    
    def __init__(self):
        self.model = CNN(classes=3, drop_prob=0.4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.loss_fn = nn.CrossEntropyLoss()
    

    def image_replacing_nan(self, images):
        nan_mask = np.isnan(images)
        means = np.nanmean(images, axis=(2, 3), keepdims=True)  
        images[nan_mask] = np.broadcast_to(means, images.shape)[nan_mask]
        clipped_images = np.clip(images, 0, 255)
        return clipped_images
        
    
    def standardising_images(self, images):
        return images / 255.0
    
    def images_labels_filter_nan(self, images, labels):
        not_nan_indices = ~np.isnan(labels)
        filtered_images = images[not_nan_indices]
        filtered_labels = labels[not_nan_indices]
        return filtered_images, filtered_labels

    def oversample_data(self, images, labels, target_label=2, target_count=300):
        target_indices = np.where(labels == target_label)[0]

        num_samples_to_add = target_count - len(target_indices)

        if num_samples_to_add <= 0:
            return images, labels

        indices_to_add = np.random.choice(target_indices, num_samples_to_add)

        oversampled_images = np.concatenate((images, images[indices_to_add]), axis=0)
        oversampled_labels = np.concatenate((labels, labels[indices_to_add]), axis=0)

        return oversampled_images, oversampled_labels

    def data_processing(self, images, labels):
        images_with_no_nan = self.image_replacing_nan(images)
        images_standardized = self.standardising_images(images_with_no_nan)
        processed_images, processed_labels = self.images_labels_filter_nan(images_standardized, labels)
        processed_images, processed_labels = self.oversample_data(processed_images, processed_labels) 
        return processed_images, processed_labels

    

    def fit(self, X, y):
        """
        Train the model using the input data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns an instance of the trained model.
        """
        X_processed, y_processed = self.data_processing(X, y) 
        X_processed = torch.tensor(X_processed, dtype=torch.float32)
        y_processed = torch.tensor(y_processed, dtype=torch.long)
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))
        ])

        dataset = DataLoaderHeler(X_processed, y_processed, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model.train()
        losses = []
        for epoch in range(55):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                loss += loss.item()

            loss = loss / len(dataloader)
            losses.append(loss)
            print ("Epoch: {}, Loss: {}".format(epoch, loss))

    def predict(self, X):
        """
        Use the trained model to make predictions.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Input data.
            
        Returns
        -------
        ndarray of shape (n_samples,)
        Predicted target values per element in X.
           
        """
        # TODO: Replace the following code with your own prediction code.
        X = self.image_replacing_nan(X)
        X = self.standardising_images(X)
        #print("X with no nan in predict:", X)
        X = torch.tensor(X, dtype=torch.float32) 
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            #print("Pred in model: ", predictions)
            return torch.argmax(predictions, dim=1)

# Load data
with open('data.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True).item()
    X = data['image']
    y = data['label']

# Split train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
def split_data(images, labels, test_size=0.2, val_size=0.25):

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels)

    # Split training data into training and validation set
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size based on remaining dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=42, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


nan_indices = np.argwhere(np.isnan(y)).squeeze()
mask = np.ones(y.shape, bool)
mask[nan_indices] = False
filtered_images = X[mask]
filtered_labels = y[mask]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(filtered_images, filtered_labels)
nan_indices = np.argwhere(np.isnan(y_test)).squeeze()
mask = np.ones(y_test.shape, bool)
mask[nan_indices] = False
X_test = X_test[mask]
y_test = y_test[mask]
#print("y_val shape: ", y_val.shape)
# Train and predict
'''
best_f1 = 0.0
best_param = {}
for lr in [1e-4, 1e-3, 5e-3]:
    for drop_prob in [0.3, 0.4, 0.5]:
        model = Model(lr=lr, drop_prob=drop_prob)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_param = {'lr': lr, 'drop_prob': drop_prob}
print("Best Parameters:", best_param)
'''
model = Model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.count_nonzero(y_test == 2))

#print(y_pred)
# Evaluate model predition
print("F1 Score (macro): {0:.2f}".format(f1_score(y_test, y_pred, average='macro')))
