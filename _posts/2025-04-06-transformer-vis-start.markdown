---
layout: post
title: "Feature Visualisation for Transformers (a first attempt)"
date: 2025-04-06 15:30:00 +0000
categories: research ai interpretability feature-vis
---

With some of my previous posts ([here](../../../2024/01/03/unreg_feature_extract.html), [here](../../../2024/01/04/reg-feature-extract.html), and [here](../../../2025/04/02/act-max.html)) looking at visualising the learned representations of Convolutional Neural Networks (CNNs), the obvious next step is to extend this to transformer-based architectures. 

We'll focus on the Vision Transformer (ViT)[^1], which is a transformer-based image classification architecture that – like the previous architectures I've experimented with before – can achieve good classification accuracy on ImageNet and other benchmark datasets. ViT uses a transformer architecture which is as close to the original[^2] as possible, which poses some issues when we move from natural language to images. The original transformer architecture expects inputs to be a finite sequence of one-dimensional tokens, which is challenging if we want to input two-dimensional images. To ameliorate this issue, the authors propose splitting the image into non-overlapping patches, flattening them, passing them through a layer that performs linear projection, taking these patch embeddings and adding a position embedding to add some positional relationship, and finally prepending with the embedding of the class (during training). This converts the image into a sequence of tokens that can then be passed to the transformer architecture ([see paper for more details](https://arxiv.org/pdf/2010.11929)). The interesting part of ViT is that it does not contain any convolution layers, so there are no kernels that we can visualise. We need to find another way to extract learned features!

According to 3b1b[^3], the interstitial, fully connected layers that occupy the space between the attention layers contain neurons, which represent questions. When we maximise the activation, we find the highest possible response to the question, which should give us some idea about what this question is!

With some minor changes to the visualisation code presented in [this post](../../../2025/04/02/act-max.html), we can get it working with transformer networks! The most significant adjustments are detailed below:
- Setting `opt_type` to `neuron` and changing the extraction process: `act = activation[:, filters]`. 
- Changing the image transformations: `vit_transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()`, which are used in the `viz_transform` composition rather than the `googlenet_transform`.

If we target a later layer in the transformer (`encoder.layers.encoder_layer_11.mlp.3`) and optimise the image, we get the following result.

![Some random kind of patterns that don't look like anything in particular](res/transformer_vis_1.png)

This image doesn't really give us many answers with respect to the 'question' the neuron may be asking. While we do see some patterns emerge, there isn't really any connection between the random bursts of something vaguely recognisable. In addition, if we optimise a second time, we get this image.

![Some random kind of patterns that don't look like anything in particular again](res/transformer_vis_2.png)

While there are some aspects of both images that are similar, there seems to be a high variance in the patterns being produced. This indicates that the method is not converging to a consistent high activation image.

One reason for this could be the nature of the activation maximisation process itself and how it interacts with the transformer architecture. The activation maximisation process developed in the previous posts works by passing in some random noise, catching the activation of a particular kernel/neuron within the classification network, traversing the gradient in a direction that causes the kernel/neuron to respond more strongly to it, and then backpropagating that change to the input image. However, in the transformer architecture, the 'context' of information within the image (extracted by the attention heads) influences the activation of the interstitial 'question' neurons. This means that when we optimise the image, not only are we changing the image, but we're also changing the context. This kind of makes the optimisation process a moving target because the image and the context influence the neuron we're trying to optimise for, and changing the image changes the context.

This gives me the impression that this is not a suitable way to explore the internal representations of transformers. I'll need to find an alternative!

---
[^1]: Dosovitskiy et al. (2020) [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929)
[^2]: Vaswani et al. (2016) [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (which has University of Birmingham alum Llion Jones as co-author)
[^3]: 3Blue1Brown [But what is a GPT? Visual intro to Transformers - Deep learning, chapter 5](https://www.3blue1brown.com/lessons/gpt)