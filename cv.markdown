---
layout: page
title: CV
permalink: /cv/
---

🧑‍💻 [GitHub](https://github.com/dfenth)
🗄️ [LinkedIn](https://linkedin.com/in/dxf-cs)
📙 [Google Scholar](https://scholar.google.com/citations?user=8aqetBUAAAAJ&hl=en)

---
## Profile 👤
---
Research Assistant at the University of Birmingham with a strong foundation in Artificial Intelligence. Over 5 years of hands-on experience developing models that deliver impactful, responsible, and secure solutions at the forefront of AI research.

---
## Education 🎓
---
### PhD Computer Science, University of Birmingham

*September 2019 - February 2025*

Completed a PhD in Computer Science with a research focus on developing adversarially robust deep neural networks for security applications. Projects included: graph-based malware detection using control flow graphs with over 50,000 nodes; ensemble-based methods employing error correction codes; and optimisation techniques for analysing network similarity. This interdisciplinary work combined machine learning and cybersecurity to address critical challenges in securing AI systems.

### BSc Computer Science, University of Birmingham

*September 2016 - July 2019*

Graduated with first-class honours (GPA 4.25)

Dissertation: “Evolutionary Methods for Generating Secure Deep Neural Networks” - Evaluated the use of multi-objective evolutionary algorithms to generate neural networks that are more robust to adversarial attacks.

---
## Employment History 💼
---

### Research Assistant, University of Birmingham

*July 2025 - Present*

Engineered a method for edit-tolerant video authentication to combat fake media. Coordinated with stakeholders to translate theoretical cryptographic techniques into practical, efficient methods for video processing capable of executing in just a few seconds. This is a UK RISE funded project scheduled for presentation in September 2025.

### Postgraduate Teaching Fellow, University of Birmingham

*September 2021 - May 2022*

Led marking initiatives by designing and implementing an automated marking system that efficiently evaluated over 600 students’ submissions in under four hours. Worked with lecturers, designing assignments to align with learning objectives and assessment criteria. The system enabled the use of practical assignments for assessment and ensured consistent and timely feedback for students.

### Postgraduate Teaching Assistant, University of Birmingham

*September 2019 - September 2021*

Facilitated the delivery of modules by leading interactive learning sessions and helping students grasp complex concepts and develop practical skills. Collaborated closely with faculty members and fellow TAs to create teaching materials, design assignments, and assess student progress, ensuring alignment with learning objectives.

---
## Publications 📃
---

### Using Reed-Muller Codes for Classification with Rejection and Recovery

*Daniel Fentham, David Parker, Mark Ryan (2023)*

Developed a novel ensemble-based classification-with-rejection and recovery method inspired by error correcting codes to improve robustness of AI models against adversarial inputs.
- Designed a new architecture (RMAggNet) integrating Reed-Muller error-correcting codes to enable both rejection and correction of adversarial inputs
- Demonstrated recovery of the true class for the majority of low-perturbation adversarial examples, outperforming existing rejection-based defences
- Showed improved robustness under adversarial training while introducing fewer natural adversaries compared to baseline methods (Thesis work)
- Proved that ensemble size scales logarithmically with the number of classes, enabling efficient deployment for large classification tasks

Presented at Foundations & Practice of Security 2023 - Published in LNCS

[Paper](https://doi.org/10.48550/arXiv.2309.06359) - [GitHub](https://github.com/dfenth/RMAggNet)


### ARASH: Video Authentication from Redactable Hash (in progress 🏗️)

*Xiao Yang, Daniel Fentham, Shize Deng, David Oswald, Mark Ryan*

Engineered a novel method for authenticating videos while enabling privacy-preserving edits.
- Translated the theoretical cryptographic protocol into a working system
- Implemented GPU-accelerated video processing using JAX, signing a 2-minute 720p video in less than 40 seconds
- Evaluated our method against existing approaches, providing guarantees of soundness and sensitivity

---
## Projects 🛠️
---

### AndroCFG

Developed a scalable pipeline to construct control flow and function call graphs from disassembled Android applications for downstream ML-based malware detection.
- Reverse-engineered and parsed Smali bytecode to generate CFGs, function call graphs, and hybrid graph representations
- Implemented function-expansion techniques around suspicious call sites to enhance structural context for classification
- Added compatibility with established malware detection frameworks (e.g., [CFGExplainer](https://ieeexplore.ieee.org/abstract/document/9833560), [MalGraph](https://ieeexplore.ieee.org/document/9796786))
- Added `dot` export for graph visualisation and analysis
- Optimised for high-throughput processing within a toolchain handling tens of thousands of APKs on the Slurm high-performance compute platform

[Related post #1](../research/gnn/malware/2026/01/21/GEMD-paper.html) - [GitHub](https://github.com/dfenth/AndroCFG)


### Pokémon RL (in progress 🏗️)

Developing a reinforcement learning agent capable of autonomously playing and winning turn-based battles in Pokémon Crystal by interfacing directly with emulator memory.
- Reverse-engineered Game Boy Color RAM to construct a controllable RL training environment
- Programmatically manipulated in-memory game state (Pokémon parties, levels, stats, moves) to generate diverse training scenarios
- Designed abstract state representations to reduce dimensionality and improve training efficiency
- Implementing and evaluating RL policies for competitive battle performance

[Related post #1](../research/ai/rl/2026/01/14/hacking-pkm.html)


### letsfoolai

Developed and deployed an interactive AI application enabling users to manipulate handwritten digit inputs to fool a classifier using model explainability feedback.
- Built and deployed a cloud-hosted web application using Google Cloud Run
- Containerised the service with Docker to enable portability and scalable deployment
- Implemented CI/CD pipeline from GitHub for automated testing and production updates
- Enhanced baseline MNIST classifier with robustness improvements to better handle user inputs
- Integrated explainability methods to visualise model sensitivity and guide user-driven adversarial exploration

[Post](../projects/ml/2026/02/28/letsfoolai.html) - [GitHub](https://github.com/dfenth/letsfool-ai) - [Website](https://letsfoolai.cloud/)


### VGSum

Designed and implemented an autonomous LLM-based system to ingest, contextualise, and summarise video game news from Bluesky.
- Integrated with the ATProtocol to extract posts, linked articles, and metadata from curated video game news accounts
- Built an agentic workflow using LangChain and LangGraph to summarise inputs and add context using tools
- Implemented external tool integration to retrieve OpenCritic scores using RapidAPI, incorporating review data into summaries
- Evaluated agent behaviour and performance using LangSmith, identifying latency and tool-calling bottlenecks

[GitHub](https://github.com/dfenth/VGSum)