# Phase-Aware Modeling of Analog Reverb

This code repository is for the article _Phase-Aware Modeling of Analog Reverb_, on review.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./src" folder, the weights of pre-trained models inside the "./weights" folder, and the audio examples in "./examples"

### Folder Structure

```
./
├── src
├── weights
└── examples
```

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets

Datasets are available [here] (https://zenodo.org/records/7044411)

Our architectures were evaluated on two analog reverb effects: 
- Plate
- Spring 


# How To Train and Run Inference 

This code relies on Python 3.9 and PyTorch.
First, install Python dependencies:
```
cd ./src
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder


# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```

```
