# First thing is to get the needed imports, for this task we need torchvision and 
# Within torchvision we need the transforms import too.
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from datasets import load_dataset
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#next we load in the image:
path_to_stft_images = '/Users/nicolasadler/Documents/MUESemester3/MusicAndAI/Homeworks_Projects/Projects/Project_0/Project_0_2/Dialect_Detector/stft_images_bottom_right_28by28'
# This line below allows to make that image into a
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
#     transforms.ToTensor()  # Convert to tensor
# ])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor()
])

stft_images = []
for filename in os.listdir(path_to_stft_images):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(path_to_stft_images, filename)
        img = Image.open(image_path).convert("L") 
        tensor_img = transform(img)
        stft_images.append(tensor_img)

print(f"Loaded {len(stft_images)} images.")
print(f"Shape of first image: {stft_images[0].shape if stft_images else 'No images loaded'}")
num_train_images = math.floor(len(stft_images) * 0.8)
num_test_images = math.floor(len(stft_images) * 0.15)
num_validation_images = math.floor(len(stft_images) * 0.05)

labels = []
for filename in os.listdir(path_to_stft_images):
    class_label = filename.split('_')[0]
    labels.append(class_label)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Class mapping:", class_mapping)
data = list(zip(stft_images, encoded_labels))

random.shuffle(data)
random.shuffle(data)
random.shuffle(data)

train_data = data[:num_train_images]
testing_data = data[num_train_images:num_train_images+num_test_images]
validation_data = data[num_train_images+num_test_images:]
from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(1) 

# Verifying that the stft_images are correctly labeled for supervised learning

# tenth_image, tenth_label = train_data[9]
# plt.imshow(tenth_image.permute(1, 2, 0))
# plt.title(f"STFT Training Image Example (Label: {tenth_label})") 
# plt.axis("off")
# plt.show()

# fifth_image, fifth_label = testing_data[4]
# plt.imshow(fifth_image.permute(1, 2, 0))
# plt.title(f"STFT Testing Image Example (Label: {fifth_label})") 
# plt.axis("off")
# plt.show()

# fourteenth_image, fourteenth_label = validation_data[13]
# plt.imshow(fourteenth_image.permute(1, 2, 0))
# plt.title(f"STFT Validation Image Example (Label: {fourteenth_label})") 
# plt.axis("off")
# plt.show()

print(f"Loaded {len(stft_images)} images.")
print(f"Shape of first image: {stft_images[0].shape if stft_images else 'No images loaded'}")
print(f"Shape of first tensor: {stft_images[0].shape}")



train_dl = DataLoader(train_data,
                    batch_size,
                    shuffle=True)
valid_dl = DataLoader(validation_data,
                        batch_size,
                        shuffle=False)
test_dl = DataLoader(testing_data,
                    batch_size,
                        shuffle=False)


for tensor_img in stft_images[:5]:
    print(tensor_img.shape)

model = nn.Sequential()
model.add_module(
    'conv1',
    nn.Conv2d(
        in_channels=1, out_channels=32,
        kernel_size=5, padding=2
    )
)
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module(
    'conv2',
    nn.Conv2d(
        in_channels=32, out_channels=64,
        kernel_size=5, padding=2
    )
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2',nn.MaxPool2d(kernel_size=2))
x = torch.ones((4,1,28,28))
print(model(x).shape)

model.add_module('flatten', nn.Flatten())
x = torch.ones((4, 1, 28, 28))
print(model(x).shape)

model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.01))
model.add_module('fc2', nn.Linear(1024, 10))

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1)== y_batch
            ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += \
                    loss.item()*y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: '
                f'{accuracy_hist_train[epoch]: .4f}')
    return loss_hist_train, loss_hist_valid, \
        accuracy_hist_train, accuracy_hist_valid

torch.manual_seed(1)
num_epochs = 100
hist = train(model, num_epochs, train_dl, valid_dl)

x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<',
    label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
#plt.show()

# pred = model(testing_data.data.unsqueeze(1) / 255.)
# is_correct = (
#         torch.argmax(pred, dim=1) == testing_data.targets
#     ).float()
# print(f'Test accuracy: {is_correct.mean():.4f}')
# fig = plt.figure(figsize=(12, 4))
# for i in range(12):
#     ax = fig.add_subplot(2, 6, i+1)
#     ax.set_xticks([]); ax.set_yticks([])
#     img = testing_data[i][0][0, :, :]
#     pred = model(img.unsqueeze(0).unsqueeze(1))
#     y_pred = torch.argmax(pred)
#     ax.imshow(img, cmap='gray_r')
#     ax.text(0.9, 0.1, y_pred.item(),
#         size=15, color='blue',
#         horizontalalignment='center',
#         verticalalignment='center',
#         transform=ax.transAxes)
# plt.show()
model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_dl:
        pred = model(x_batch)
        all_preds.append(torch.argmax(pred, dim=1))
        all_labels.append(y_batch)

# Flatten the lists of predictions and labels
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Compute test accuracy
is_correct = (all_preds == all_labels).float()
print(f'Test accuracy: {is_correct.mean().item():.4f}')