---
layout: post
title: "Creating a training and validation split for torchvision datasets"
date: 2024-4-09 16:04:00 +0000
categories: research notes pytorch

---

I've had an annoying issue with some of the [torchvision datasets](https://pytorch.org/vision/stable/datasets.html) in that they don't split the training and validation data. I was trying to decide on the best solution to this issue today and decided to ask ChatGPT (since this is something we can verify!).

The solution it proposed is below:

```python
trainval_dataset = torchvision.datasets.OxfordIIITPet(
    root='/tmp', 
    split='trainval', 
    download=True, 
    transform=transform
    )
testing_dataset = torchvision.datasets.OxfordIIITPet(
    root='/tmp', 
    split='test', 
    download=True, 
    transform=transform
    )

val_split = 0.2
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(range(len(trainval_dataset)), test_size=val_split, random_state=8208)
training_dataset = torch.utils.data.Subset(trainval_dataset, train_indices)
validation_dataset = torch.utils.data.Subset(trainval_dataset, val_indices)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

Which seems to work quite nicely! I like the fact that you can shuffle the indices to create different training and validation sets each time (and can set it with a seed). 
