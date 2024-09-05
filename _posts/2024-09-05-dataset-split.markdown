---
layout: post
title: "Train-test splits and K-fold CV"
date: 2024-9-4 11:58:00 +0000
categories: research notes pytorch

---


[Scikit Learn](https://scikit-learn.org/stable/index.html) is a handy python library for (among many other things) train-test splits and K-fold cross-validation, so it will be featured extensively throughout the post!

## Train/Test Split
The training/testing split of data is crucial in the AI/ML field, but if you're using standard datasets (e.g. MNIST, CIFAR-10, etc.) which are included in your AI/ML framework of choice, then you don't really need to worry about it too much. The libraries (PyTorch, TensorFlow etc.) usually handle this split for you:
```python
train_mnist = torchvision.datasets.MNIST('/tmp', train=True, download=True)
test_mnist = torchvision.datasets.MNIST('/tmp', train=False, download=True)
```
The code above uses `torchvision` to load the MNIST datasets, and includes a handy boolean flag: `train` so you can specify if you want the training or testing dataset.

With this nicely abstracted approach to loading standard datasets, things get a little more complicated if you're [loading a custom dataset from a directory](../../09/04/custom-dataset.html)! However, Scikit Learn can make the train/test split much easier for us with the `sklearn.model_selection.train_test_split` function ([docs here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#train-test-split)).

Assuming we have our input data and labels in two separate python lists, we can split the data with the following code:
```python
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(
	input_data,
	labels,
	train_size=0.8,
	random_state=0,
	shuffle=True,
	stratify=labels
)
```
We pass in the input data and labels, along with the `train_size` (percentage of the data elements to use for training), the `random_state` (a seed for shuffling if we want reproducibility), `shuffle` (the option to shuffle the data) and `stratify` (which allows us to keep the ratio of classes across the training and testing datasets). The stratify option is a really useful one to have since it means that we can be sure that the ratio of the classes is consistent between training and testing data so we can test in a realistic environment. This is important when dealing with classes with very few samples (imagine a 90-10 split for a binary classifier). If we're not using a stratified approach, we can end up with 0 instances of one of the classes in the test set, which can skew our testing results and make them look better than they actually are!

From here, we can pass the `train_X`, `test_X`, `train_y`, and `test_y` results into a torch Dataset and continue as we would with any other dataset!

## K-fold Cross-Validation
Scikit Learn also gives us a convenient way to perform K-fold Cross-Validation, which is a really useful way to improve model performance if we have a small dataset. K-fold Cross-Validation essentially allows us to improve the effectiveness of training on a small dataset by repeating the training process multiple times on different subsets of the data. The training data is split into $k$ folds, with one fold held out for validation. We train on all of the other folds and validate on the holdout. We then change which fold is held out and repeat the process until each fold has been held out once. 

If we have our train and test datasets (which are of type `Dataset` from `torch.utils.data.Dataset`), then we can ignore the test dataset (we want to keep our testing dataset isolated so we can get a reliable read on how the model will perform on real data) and focus on the training dataset.

```python
from sklearn.model_selection import KFold

for epoch in range(MAX_EPOCHS):
	kf = KFold(n_splits=5, shuffle=True)
	kf.get_n_splits(train_dataset)
	
	folds = kf.split(train_dataset)

	for train_idxs, val_idxs in folds:
		train_srs = torch.utils.data.SubsetRandomSampler(train_idxs)
		train_dataset_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=32,
			sampler=train_srs
		)

		losses = []
        opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

		for input, labels in train_dataset_loader:
			# Train as normal!

		val_srs = torch.utils.data.SubsetRandomSampler(val_idxs)
        val_dataset_loader = torch.utils.data.DataLoader(
	        train_dataset, 
	        batch_size=32, 
	        sampler=val_srs
		)

		for input, labels in val_dataset_loader:
			# Validate as normal!
```
In the code above, we create $k$ (5) new folds for each epoch of training. The `folds` variable consists of two lists; the first is the list of training indices, and the second is the validation indices. For both the training and validation indices, we create a `SubsetRandomSampler`, which allows us to use these indices to load a dataset via `DataLoader`. From here, we can train and validate as we usually would!
