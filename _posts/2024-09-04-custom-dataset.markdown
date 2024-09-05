---
layout: post
title: "Loading a custom dataset"
date: 2024-9-4 11:58:00 +0000
categories: research notes pytorch

---

There are a few ways to load a custom dataset in PyTorch. Here, I'll focus on (I think) the most common type of dataset you may need to manually load, that is, all of the images in a single directory with a `csv` that contains the image name and class the image belongs to.

Starting with the imports:
```python
import os
import glob
import torch
import torchvision
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
```
We can then check if the dataset location actually exists:
```python
dataset_path = "some_random_path"
labels_path = "some_random_labels_path.csv"
assert os.path.isdir(dataset_path)
```
We can define a function to make the initial dataset loading more contained. The code below loads the `csv` containing the classes and makes two lists, one containing the images and the other containing the labels. The images are loaded as PyTorch tensors using torchvision! Finally, we can create training and testing splits using scikit learn's `train_test_split`. 

```python
def load_initial_dataset(data_dir, class_path):
     # Get the dataset image (loaded as a pytorch tensor) and labels
    images, image_labels = [], []
    with open(class_path, "r") as sig_file:
        for line in sig_file.readlines()[1:]: # Skip the header
            line = line.replace("\n", "")
            img_path, img_label = line.split(",")
            image_path = glob.glob(os.path.join(data_dir, img_path + ".png"))
            images.append(transform(torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(image_path))))
            image_labels.append(img_label)
    
    # Create the dataset using scikit learn train_test_split
    train_X, test_X, train_y, test_y = train_test_split(
        images, 
        image_labels, 
        train_size=0.8, 
        random_state=8208, 
        shuffle=True, 
        stratify=image_labels
    )

    return train_X, train_y, test_X, test_y
```
Next we can define a `Dataset` class which we can use to handle the dataset:
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.data = X
        self.labels = y
        self.transform = transform # Any transform we want to perform on data loading!

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        image = self.transform(image) if self.transform else image
        return image, label

    def print_dataset_stats(self):
        valid = sum(self.labels)
        invalid = len(self.labels) - valid
        print("Dataset consists of {} samples".format(
            len(self.labels)))
```
That's all! We can invoke the custom dataset loading with the following code:
```python
some_transforms = v2.Compose([
    v2.RandomCrop((224, 224)),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
        
train_X, train_y, test_X, test_y = load_initial_dataset(dataset_path, labels_path)
train_dataset = CustomDataset(train_X, train_y, transform=some_transform)
test_dataset = CustomDataset(test_X, test_y, transform=some_transform)

print("Training stats:")
train_dataset.print_dataset_stats()
print("Testing stats:")
test_dataset.print_dataset_stats()
```

