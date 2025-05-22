#!/usr/bin/env python
""" fine tune MobileNetV3Large to classigy images types
full ml lifecycle: data split, preprocess, augment, train, eval, save model
"""
# import libraries
import os
import shutil
import random
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# check ig gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/"
split_dir = "data/splits/"
splits = ["train", "val", "test"]
splits_ratios = [0.7, 0.15, 0.15]
classes = ["meme", "nonMeme"]

# create split dir
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)
for cls in classes:
    img_dir = os.path.join(data_dir, cls)
    images = os.listdir(img_dir)
    # shuffle imgs so we don't get same imgs in training and testing
    random.shuffle(images)
    total = len(images)
    # calculate training img set (example total=100 and ratio 70 so train_end
    # = 70)
    train_end = int(splits_ratios[0] * total)
    val_end = train_end + int(splits_ratios[1] * total)
    splits_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }
    for split in splits:
        for img in splits_data[split]:
            src = os.path.join(img_dir, img)
            # create destination path like ../splits/train/meme/meme123.jpg
            dest = os.path.join(split_dir, split, cls, img)
            # copy img
            shutil.copy(src, dest)
print("data split done")
# data transformation (resize to expected model size 448*448 and randomly change
# brightness and contrast for train imgs to make model sees different lightings and convert
# them to a pytorch tensor:numbers model can use also normalize image pixels to same scale
# the original model was tartrained on)

train_tf = transforms.Compose([transforms.Resize((448, 448)),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])
                               ])
# now val test data same as train in resizing and normalization but
# without brightness/contrast
test_tf = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# load img folder and apply transformation
train_data = datasets.ImageFolder(
    os.path.join(
        split_dir,
        "train"),
    transform=train_tf)
val_data = datasets.ImageFolder(
    os.path.join(
        split_dir,
        "val"),
    transform=test_tf)
test_data = datasets.ImageFolder(
    os.path.join(
        split_dir,
        "test"),
    transform=test_tf)
# create batch (32)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# load the pretrained model
model = models.mobilenet_v3_large(pretrained=True)
# freeze all layers only the final classification head
for param in model.parameters():
    param.requires_grad = False
# replace final layer with a new layer having outputs same as our classes
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, len(
        train_data.classes))
# move model to gpu for faster training
model = model.to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer adam
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
# repeat training 5 rounds
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        # get model predictions
        outputs = model(imgs)
        # compare model to true labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} loss: {total_loss / len(train_loader):.4f}")
# test accuracy
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    # loop over test img
    for imgs, labels in test_loader:
        # move data to gpu
        imgs, labels = imgs.to(device), labels.to(device)
        # get predictions
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
# accuracy
accuracy = accuracy_score(all_labels, all_preds)
# save trained model weights
torch.save(
    model.state_dict(),
    "img_type_classification/mobilenetv3_scene_classifier.pth")
print("")
print("classification report")
print(classification_report(all_labels, all_preds, target_names=classes))
