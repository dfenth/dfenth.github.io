---
layout: post
title: "Graph Embedding for Malware Detection - A failed paper"
date: 2026-01-21 17:00:00 +0000
categories: research gnn malware
usemathjax: true

---

⚠️⚠️⚠️
> This piece of work was done fairly early on in my PhD. This was back when I was focusing on Geometric Deep Learning and trying to change the world of malware detection with some novel approaches to application representation. The gist of what I wanted to do was to break an app down into basic-blocks and analyse the geometry of the code graph. My feeling was that there may be some unique geometry associated with malware that we can detect, and due to this being a more fundamental part of the structure of a program, it may be more resilient to concept drift. This didn't end up working out. I worked on this in years 2 and 3 of my PhD and with about a year left to write my thesis I had a massive training pipeline and an AI model that took about a month to train. The approach wasn't working and I didn't know which part was the problem. With bugs to fix and only a handful of opportunities to actually train the AI model, I would have been under a lot of pressure to get something working for my thesis (if it was even possible). I decided to cut my losses and change direction. I'll be making a series of posts about this failed project. Even though it didn't work, I'm still really proud of what I produced.

⚠️⚠️⚠️

---

## Abstract

The development of malware detection methods in the Android ecosystem has benefited from a huge amount of research in the past few years. Many Machine Learning based approaches have reported incredible performance, but are challenged by real world deployment for two primary reasons. The first is that they are susceptible to dataset drift, occuring as malware evolves to avoid detection which reduces the model performance as a function of time. The second is the lack of transparency in the classifications which inhibits their use in safety and security critical applications. 

In this paper we attempt to tackle these two concerns using Graph Neural Network based classifiers and Explainable AI techniques. While there have been many papers proposing Graph Neural Networks as malware classifiers, none have explored the robustness of them to dataset drift, a crucial component to determining their ability to operate effectively in the real world. To explore this point we analyse the ability of Graph Neural Network based classifiers to retain high classification values in the presence of dataset drift. We also explore the use of Explainable AI in analysing the salient graph features which contribute to classification. This provides deep insights into characteristics of malware by providing human interpretable features which increase the verifiability of the classification model and thus the trustworthyness in safety and security critical applications.

---

## Introduction

The Android Operating System (OS) and its many variants are collectively the most popular in the world accounting for approximately 43% of all computer devices[^1]. This popularity has naturally made it a prime target for malware as adversaries attempt to extract personal information from its many users. Malware typically infects Android devices through the user downloading a malicious application (app) from one of the many app stores available. This has led to a wealth of research focused on classifying apps as benign or malicious based on their properties. Many of these classifiers report high $F_1$, precision and recall values which appears to indicate that this is a solved problem. However, work conducted by Pendlebury et al.[^2] has shown that the reported $F_1$, precision, and recall values often do not tell the full story.

All malware classifiers will suffer from dataset drift. Dataset drift captures many different shifts that can occur in the dataset over time, but it encapsulates the idea that as time progresses the data a classifier is expected to operate on becomes further from the distribution of data it is trained on. Concept drift is of particular interest to us due to the nature of malware, where malicious programs are constantly changing and evolving to avoid detection by classifiers. Testing classifiers in the presence of concept drift is a key evaluation metric which is missing from many of the proposed malware classification methods which presents challenges when it comes to deploying the models in a real world situation.

Approaches to malware classification in the Machine Learning (ML) community have focused on extracting certain properties of the apps which the authors believe would be useful to the categorisation task. Classifiers such as Drebin[^3] create feature vectors for each app consisting of metadata such as requested hardware components and permissions along with code information such as suspicious API calls and network addresses. Competing ideas in the malware classification space come from Graph Neural Networks (GNNs). Methods such as MAGIC[^4] and MalGraph[^5] argue that we can create more powerful classifiers by representing programs as graphs and use the resulting structure as a feature in addition to features that can be extracted from the code itself. Programs have many natural graph representations, such as Function Call Graphs (FCGs), Control Flow Graphs (CFGs) and Data Flow Graphs (DFGs), therefore this approach to classification has gained traction in the malware detection community. The results reported by many of the GNN malware detection papers show performance competitive with the non-graph variants, however, there is (to our knowledge) no evaluation of GNN methods with respect to concept drift.

We can intuitively reason that GNN approaches to malware classification may be more robust than non-GNN approaches by considering the data we use for classification. Non-GNN classifiers rely on app features which are generally more susceptible to changes in time. For instance both Drebin[^3] and MaMaDroid[^6] create feature vectors based on API calls (or families of API calls) which can change over time as Android updates the APIs available to developers. On the other hand while GNN methods include API information, this is often augmented by structure information in the form of the program graphs. It is less likely that these graph structures will display significant changes over time which may make them more robust to concept drift.

Explainable Articificial Intelligence (XAI) is a tool that can be used to increase model transparency by highlighting features of the input that are salient to the classification. In the field of malware research, where the safety and security of the system are highly important, this can increase the confidence that end-users have in the model, allowing results to be verified in that we can ensure that the features that are being used for classification are appropriate. In this paper we use an implementation of SubgraphX[^7] to inspect the features that are being used for malware classification which are (hopefully) predominantly subgraph structures. This (hopefully) supports our hypothesis that subgraph structures can be used as robust and reliable features which are characteristic of malware.

*Our contributions*: This paper evaluates the effect that concept drift has on GNN based malware classifiers in the Android ecosystem. We (hopefully) show that GNN classifiers are more robust to the effects of concept drift, retaining high scores for many months that succeed the training data. We use SubgraphX[^7] to confirm that this behaviour is due to the effective use of the program graph structure which augments the classification process with rich data which is slower to change compared to specific code based features (such as API usage, URL schemes etc.) which are predominantly used by non-GNN based malware classifiers.

---

## Related work

### Malware Classification

The automated detection of malware using ML methods has been an area of active research for many years, particularly in the space of Android applications. Popular Android malware classifiers such as Drebin[^3] and MaMaDroid[^6] apply traditional ML techniques to feature vectors they extract from apps. Drebin focuses on collecting features such as requested hardware access, system permissions, inter/intra-process communications and interfaces to the OS along with restricted API calls and app permissions. This information is then embedded and classified using a Support Vector Machine (SVM). MaMaDroid considers the app program graph in the form of a package or family API call graph extracted from the program binary. The call graph is used to create a Markov chain which models the probability of a transition occurring between API calls. This Markov chain is then used to create a feature vector where entries are the probability of transitioning between the API calls. This is then reduced using Principal Component Analysis (PCA) to an embedding space for classification.

These approaches produced strong $F_1$, precision and recall values, however, they were later shown by Pendlebury et al.[^2] to suffer from concept drift which showed that their effectiveness in real-world applications is limited. This finding could be due to the reliance of these methods on API call analysis which can be volatile between OS updates.

GNNs have been gaining in popularity in the malware classification space, primarily because of their expressive nature of their inputs which allows us to leverage structure of data in the classification process. The most intuitive representation of program data is a program graph which was leveraged by a paper proposing MAGIC[^4]. MAGIC focused on classifying benign and malicious programs from Windows binaries using GNNs. The graph representation of the binaries are created by decompiling the programs into assembly, then creating basic-blocks from the instructions. The graph consists of a basic-block as the graph node which contains summary information about the instructions contained within them, with edges representing the connectivity between the basic-blocks in the code structure. This graph is then fed into a Deep Graph Convolutional Neural Network (DGCNN)[^8] with graph reduction layers such as SortPool[^8] before classification which consists of a multi-layer perceptron.

A different approach to the Graph based classification method, which demonstrates the expressive nature of the technology, can be seen in a paper which introduces MalGraph[^5]. MalGraph uses a hierarchical graph representation to perform classification using GNNs. At the high level of the hierarchy MalGraph defines an inter-function FCG which details the high level function interactions, and the lower level hierarchy is populated with intra-function CFGs of local functions. This ultimately creates a multi-modal graph where nodes can be local functions which have been processed by a GNN resulting in an embedding to a local function space, or an external function node which is simply one-hot encoded based on the function name. The multi modal FCG is passed to another GNN, followed by a multi-layer perceptron for classification.

One recent application of these graph methods to the Android malware space was presented in a paper by Hei et al. which proposes Hawk[^9]. Hawk uses a Heterogeneous Information Network (HIN) which creates a graph of heterogeneous nodes from features such as program APIs, permissions, interfaces etc. with apps being added to the graph with connectivity based on the features they have (e.g. an app node will be connected to a permission node if an app has that permission). A homogeneous graph is then extracted which just contains the app nodes. This is then passed to a variant of the Graph Attention Network (GAT)[^10] for classification.

While the graph approaches detailed here all report a strong ability to differentiate malware from goodware on their datasets, there has been (to our knowledge) no thorough analaysis of these methods with respect to concept drift, an issue this paper hopes to change.

### Explainable AI

XAI is an invaluable technology in situations where safety and security are of critical importance. In this paper XAI is used to determine features which are salient in the malware classification process. Several papers address the application of XAI to GNNs such as GNNExplainer[^11], GNN-GI and GNN-LRP[^12] and PGExplainer[^13] which all focus on structural node, edge, or node feature explanations of the graphs. Our focus will be on SubgraphX[^7] which specifically uses subgraphs for explanation which, the authors argue, provide a more human interpretable and useful explanation of the classifications. 

The application of XAI methods to Graph-based malware classification has been carried out by Herath et al.[^14] with CFGExplainer.

---

[^1]: Source: [statcounter (April 2022)](https://gs.statcounter.com/os-market-share)
[^2]: Pendlebury et al. (2019) [TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time](https://www.usenix.org/system/files/sec19fall_pendlebury_prepub.pdf)
[^3]: Arp et al. (2014) [DREBIN: Effective and Explainable Detection of Android Malware in Your Pocket](https://media.telefonicatech.com/telefonicatech/uploads/2021/1/4915_2014-ndss.pdf)
[^4]: Yan et al. (2019) [Classifying Malware Represented as Control Flow Graphs using Deep Graph Convolutional Neural Network](https://drive.google.com/file/d/1gR_sa61DN3SkwTi2PC7B9V0ipofis3tU/view)
[^5]: Ling et al. (2022) [MalGraph: Hierarchical Graph Neural Networks for Robust Windows Malware Detection](https://ieeexplore.ieee.org/document/9796786)
[^6]: Mariconti et al. (2016) [MaMaDroid: Detecting Android Malware by Building Markov Chains of Behavioural Models](https://arxiv.org/pdf/1612.04433)
[^7]: Yuan et al. (2021) [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/pdf/2102.05152)
[^8]: Zhang et al. (2018) [An End-to-End Deep Learning Architecture for Graph Classification](https://ojs.aaai.org/index.php/AAAI/article/view/11782)
[^9]: Hei et al. (2024) [Hawk: Rapid Android Malware Detection Through Heterogeneous Graph Attention Networks](https://ieeexplore.ieee.org/document/9524453)
[^10]: Veliˇckovi´c et al. (2018) [Graph Attention Networks](https://arxiv.org/pdf/1710.10903)
[^11]: Ying et al. (2019) [GNNExplainer: Generating Explanations for Graph Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Paper.pdf)
[^12]: Schnake et al. (2022) [Higher-Order Explanations of Graph Neural Networks via Relevant Walks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9547794)
[^13]: Luo et al. (2020) [Parameterized Explainer for Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf)
[^14]: Herath et al. (2022) [CFGExplainer: Explaining Graph Neural Network-Based Malware Classification from Control Flow Graphs](https://ieeexplore.ieee.org/document/9833560)

