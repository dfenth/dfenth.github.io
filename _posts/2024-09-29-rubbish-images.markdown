---
layout: post
title: "Rubbish Images"
date: 2024-09-28 18:11:00 +0000
categories: research ai adversarial-attacks out-of-distribution
---

A really interesting way to demonstrate how broken many Neural Networks are is to think about *'rubbish inputs'*, which were formalised by *Nguyen et al*[^1]. The authors focused mainly on image classification, but this applies to pretty much any neural network which doesn't have some form of out-of-distribution detection.

The idea here is that most (non-rejection) neural nets *must* classify whatever you give them, regardless of whether it's within the bounds of what they know. This can lead to some interesting results, which are very strange to human observers. The images below show an Inception-V3[^2] network (a fairly standard image classification network used as a base for many other image classifiers) being very certain of the classes, even though the images are just noise to us humans.
![Images of noise with classifications very certain they're images of a zebra, envelope and bagel respectively](res/rubbish.png)

The code to produce these images is very simple.
```python
def optimise_rubbish(model, target_class, iterations, eps, device):
	# Set the model to eval mode so we don't accidentally change
	# model parameters (I've been there...)
	model.eval()

	# Initialise a random noise image to optimise
	noise_input = torch.rand(size=(1, 3, 224, 224), device=device, requires_grad=True)

	# Target the noise image in the optimisation
	opt = torch.optim.Adam(params=[noise_input], lr=eps)
	loss_fn = torch.nn.CrossEntropyLoss()

	# Start the optimisation 
	# (can usually get to 0 loss within 100 iterations)
	for it in range(iterations):
		opt.zero_grad()
		predictions = model(noise_input)
		loss = loss_fn(predictions, target_class)
		loss.backward()
		opt.step()

		if it % 10 == 0:
			print("Iteration {:03} - Loss: {:.4f}".format(it, loss.item()))

	return noise_input.detach().cpu()
```

This essentially initialises a random noise image, then passes it to the model (the neural net) and finds out what class the model thinks it is. It then calculates a loss to determine the discrepancy between what the model thinks the image is and what we *want* it to think the image is. With this loss, we can optimise the image in a way which pushes it towards the class we want! This is very similar to basic adversarial attacks such as BIM[^3], we're just optimising the image directly. Within a few iterations, we can get to the desired class and then have a very confidently classified rubbish image!

The full code is below:
```python
import torch
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

imagenet = torchvision.models.inception_v3(weights='DEFAULT')
imagenet = imagenet.to('cuda')
image_transforms = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()

# Load the classes from an external file
imagenet_classes = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]

def optimise_rubbish(model, target_class, iterations, eps, device):
	target_class_idx = None

	# This section allows us to reference imagenet classes by name and
	# recover the index
	for idx, cl in enumerate(imagenet_classes):
		if target_class == cl:
			target_class_idx = torch.tensor(idx).unsqueeze(dim=0).to(device)
			break
	else:
		raise ValueError("Class not found")

	model.eval()

	noise_input = torch.rand(size=(1, 3, 224, 224), device=device, requires_grad=True)
	
	opt = torch.optim.Adam(params=[noise_input], lr=eps)
	loss_fn = torch.nn.CrossEntropyLoss()
	
	for it in range(iterations):
		opt.zero_grad()
		predictions = model(image_transforms(noise_input))
		loss = loss_fn(predictions, target_class_idx)
		loss.backward()
		opt.step()
		
		if it % 100 == 0:
			print("Iteration {:03} - Loss: {:.4f}".format(it, loss.item()))

	return noise_input.detach().cpu()

rubbish = optimise_rubbish(imagenet, "envelope", 100, 1e-1, device)
```

This example shows a side of classifiers that isn't often discussed. To combat this, approaches which perform some kind of out-of-distribution detection must be used to ensure the model doesn't make overconfident misclassifications!

AI can do some really impressive things, but most models in production are very task-specific, and the input must conform to the training data distribution for them to behave as expected. Once you start throwing *rubbish* at them, the seams show quite quickly.

[^1]: Nguyen et al. 2015 ["Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images"](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
[^2]: Szegedy et al. 2015 ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/pdf/1512.00567)
[^3]: Kurakin et al. 2018 ["Adversarial examples in the physical world"](https://arxiv.org/pdf/1607.02533)