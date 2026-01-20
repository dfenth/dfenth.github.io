---
layout: post
title: "Old notes on Graph Neural Networks"
date: 2026-01-20 17:00:00 +0000
categories: research notes gnn
usemathjax: true

---

Graph Neural Networks can be expressed most simply as:

$Z = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW+b$

Where:

$\hat{A} = A+I$, $\hat{A} \in \mathbb{R}^{N\times N}$ is the adjacency matrix ($N$ is the number of graph vertices/nodes)

$\hat{D}\_{ii}=\sum\_j^N \hat{A}\_{ij}$, $\hat{D} \in \mathbb{R}^{N \times N}$ is the degree matrix

$X \in \mathbb{R}^{N\times F}$ is the feature matrix ($F$ is the number of features we have for each vertex)

$W \in \mathbb{R}^{F\times C}$ is the learned weight matrix ($C$ is the number of weight channels we choose to have)

$b \in \mathbb{R}^{F \times C}$ is the learned bias matrix

Note that the weight and bias part of the above equation is often shortened to $\Theta$ were $\Theta = W+b$.

We're likely to want to chain multiple graph convolutions together in a ML system, so we re-write the equation above as:

$Z^{(l+1)} = \sigma \left( \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}Z^{(l)}\Theta^{(l)} \right)$

Where $l$ denotes the layer and $\sigma$ is a non-linear activation function (usually a sigmoid or ReLU) and $Z^{(0)} = X$.

What we can see here is that each layer of the graph convolution returns another value of $X$, which means we're operating over the features of the vertices of the graphs. The inclusion of  $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ shows we're using some normalised information from the topology of the network to influence this convolution, with this all being weighted by the learned parameters $\Theta$. It's important to note that $Z$ changes shape as the convolution continues. Initially since $Z^{(0)} = X$, $Z^{(0)} \in \mathbb{R}^{N \times F}$ and later as we multiply by $\Theta \in \mathbb{R}^{F \times C}$ then $Z^{(l)} \in \mathbb{R}^{N \times C}$ which is a matrix of size defined by the number of vertices by the number of weight kernels (each column can be thought of as a way of assigning a value to each node in the graph). $W^{(0)} \in \mathbb{R}^{F \times H}$ (an input to hidden layer with $H$ feature maps) with $W^{(1)} \in \mathbb{R}^{H \times C}$.

Practically, since $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ is a constant for graph convolution we can calculate it once and forget about it: $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$

We can then reduce the above equation to a much cleaner form:

$Z^{(l+1)} = \sigma \left( \tilde{A}Z^{(l)}\Theta^{(l)} \right)$

We can easily see how they'll all chain together:

$Z = \sigma^{(L)}(\dots\sigma^{(1)}(\tilde{A} \sigma^{(0)}(\tilde{A}X\Theta^{(0)})\Theta^{(1)})\dots\Theta^{(L)})$

### Looking at $\Theta$

Since $\Theta \in \mathbb{R}^{F \times C}$ then it doesn't depend on $N$ (the number of vertices). In fact, it doesn't rely on a specific graph topology at all! This means that we have a graph invariant parameter (which was the focus of much of the early GNN research) which means we can use this learned parameter on many different graphs. This gives credibility to the idea that we can change both the graph and the graph data every training example and still be able to learn a $\Theta$ that can be useful to us!

