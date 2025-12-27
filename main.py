import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib as Path
import os

gpu = False

# verify if GPU parallel calculus is possible
if(torch.cuda.is_available()):
    print(f"GPU available: {torch.cuda.get_device_name()}")
    gpu = True
else:
    print("No GPU available")
    
    import shutil
import kagglehub

dataset_url = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR = "chest_xray/chest_xray"

def load_data():
    """
    Load original pneumonia images data from the Kaggle's dataset. 
    """
    # if it finds the data does nothing otherwise it installs it --> cannot pass the dataset to Github for dimension limitations
    if os.path.exists(DATA_DIR):
        print(f"Data found in {DATA_DIR}")
    else:
        # Download latest version --> kagglehub stores the dataset in a sort of cached memory so you need to copy it in local
        path = kagglehub.dataset_download(dataset_url)
        
        shutil.copytree(path, ".", dirs_exist_ok=True)

        print("Path to dataset files:", path)
        
load_data()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

RESIZE_IMG = (128, 128) # resize img from (224,224) -> (128, 128)
datasets_names = ["train", "val", "test"]

# we create a set of transformations -> operations that will be applied to each img of the dataset for preprocessing
data_transforms = {
    x: transforms.Compose([
        transforms.Resize(RESIZE_IMG), # resizes the image to the desired dimension
        transforms.Grayscale(num_output_channels=3), # the paper uses three channels for these images
        transforms.ToTensor(), # convert all the values from range [0, 255] -> [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], # Standard ImageNet normalization
                             [0.229, 0.224, 0.225]) # a quanto pare sono valori standardizzati usati per trainare modelli (si potrebbe pensare di rimuoverli)
    ]) for x in datasets_names
}

# build a dictionary with three keys for each dataset we have (train, val, test)
# it also applies the transformations defined before
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in datasets_names
}

class_names = image_datasets["train"].classes
print(class_names) # print the two classes we are labeling

# data on the sizes of the single datasets 
dataset_sizes = {x: len(image_datasets[x]) for x in datasets_names}

# printing the number of elements for each dataset.
for x, y in dataset_sizes.items():
    print(f"Dataset {x} has {y} elements.")
    
    mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """ 
        Given a tensor in input this function returns the corresponding image
        This function is not necessary if in the transformations we do not apply normalization
    """
    img = tensor.numpy().transpose((1, 2, 0)) # change the order of variables in order to get (width, heigth, channels)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    return img

# access the first ten images of the train dataset and print it
fig, axis = plt.subplots(2, 5, figsize=(20, 8))
axis = axis.flatten()  # make it an array instead of a matrix
for i in range(10):
    image_tensor, img_label = image_datasets["train"][i]
    
    img = denormalize(image_tensor)
    class_name = class_names[img_label] # if 0 will be 'NORMAL', IF 1 'PNEUMONIA'
    
    axis[i].imshow(img)
    axis[i].set_title(f"Class: {class_name}")
    
BATCH_SIZE = 32
# initializing dataloaders for parallel calculus with GPU
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for x in datasets_names
}
# we access images in batches with an iterator
batch_images, batch_labels = next(iter(dataloaders["train"]))

print(batch_images.shape) # prints [batches, channels, width, heigth]
print(len(batch_images))

# another plot of the images, this time shuffled (lo sto facendo solo per prenderci la mano con i plot)
fig, axs = plt.subplots(4, 8, figsize=(20,12))
axs = axs.flatten()
for i in range(BATCH_SIZE):
    img = denormalize(batch_images[i])
    label_name = class_names[batch_labels[i]]
    axs[i].imshow(img)
    axs[i].set_title(f"{label_name}")
    axs[i].axis("off")
    
# here I define a list of augumentation (I just specified the most important one cited in the paper)
data_augumentations = [
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2), #setted to 20% as the paper suggest
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))
]

# here I update the train dataset with the augumentations
data_transforms["train"] = transforms.Compose([
    transforms.Resize(RESIZE_IMG), 
    *data_augumentations, # unpack operators takes the element of the list and puts them inside another one
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

image_datasets["train"] = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms["train"]) # just modify the train one

# update dataloaders with the new augmented set
dataloaders["train"] = DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

batch_images, batch_labels = next(iter(dataloaders["train"]))

# last images plot ( I SET THE BRIGHTEN AUGUMENTATION TO 50% JUST TO MAKE THE RESULTS VISIBLE BUT THAT HAS TO BE CHANGED TO 20%)
fig, axs = plt.subplots(4, 8, figsize=(20,12))
axs = axs.flatten()
for i in range(BATCH_SIZE):
    img = denormalize(batch_images[i])
    label_name = class_names[batch_labels[i]]
    axs[i].imshow(img)
    axs[i].set_title(f"{label_name}")
    axs[i].axis("off")
    
#apply oversampling for class balance with SMOTE tecnique

from imblearn.over_sampling import SMOTE 
from torch.utils.data import TensorDataset

#count occurrences
train_label = image_datasets['train'].targets
count_normal = train_label.count(0)
count_pneumonia = train_label.count(1)

print(f"Count Normal: {count_normal}")
print(f"Count Pneumonia: {count_pneumonia}")

#we have to flatter images because SMOTE works with "tables" of numbers--> so we have to transform images into number vectors 

#temporary lists
train_data_flat = []
train_labels = []

for img, label in image_datasets['train']:
    train_data_flat.append(img.numpy().flatten()) #numpy transform img in array, flattern transorm it in one dim array
    train_labels.append(label)

train_data_flat = np.array(train_data_flat)
train_labels = np.array(train_labels)

print(f"Actual dimension: {train_data_flat.shape}")

smote = SMOTE(random_state=42)

img_resampled, label_resampled = smote.fit_resample(train_data_flat, train_labels) #fit resample learn from datas and make new datas similar 

print(f"Now we have: {len(label_resampled)} images.")

#check the balance
unique, counts = np.unique(label_resampled, return_counts=True)
print(counts)

#at the end we have to re-convert arrays of number in images for ur CNN

img_resampled_tensor = torch.tensor(img_resampled).float().view(-1, 3, 128, 128) 
label_resampled_tensor = torch.tensor(label_resampled).long()

train_dataset_smote = TensorDataset(img_resampled_tensor, label_resampled_tensor)

dataloaders["train"] = DataLoader(train_dataset_smote, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ================================================================================

print(batch_images.shape) # print again the tensor shape

import torch.nn as nn

pipeline = nn.Sequential(
    nn.Conv2d(3, 32, (3, 3), padding=0),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1, inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(32, 64, (3, 3), padding=0),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.1, inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2), 

    nn.Conv2d(64, 128, (3, 3), padding=0),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.1, inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(), # Flatten the output when the pipeline is finished
    
    nn.Linear(32768, 128)
    
)

# ======================================================================

device = torch.device("cuda" if gpu else "cpu")
pipeline.to(device)

pipeline.eval()

features = []
for batch_idx, (images, labels) in enumerate(dataloaders["train"]):
    images = images.to(device) # move data to CPU/GPU
    output = pipeline(images)
    features.append(output.cpu())
    print(f"Batch n.{batch_idx + 1}: Processed {len(images)}")
    
# =======================================================================

# apply our data to the Gated Recurrent Unit (GRU)
bigru = nn.GRU(
            input_size= 128,
            hidden_size= 256,
            num_layers= 2,
            dropout= 0.3,
            bidirectional=True,
            batch_first=True
        )

for feature in features:
    feature = feature.unsqueeze(1).repeat(1, 25, 1)
    
    feature, _ = bigru(feature)
    print(feature.shape)

projections = nn.Sequential(
    nn.Linear(512, 128),
    nn.Linear(128, 64)
)

for feature in features:
    feature = projections(feature)
    print(feature.shape)




