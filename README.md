# PARAMETRIC GENERATION OF IMPULSE RESPONSES FOR AUDIO EFFECTS

This code repository is for the article _Parametric Generation of Impulse Responses for Audio Effects_, on review.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./src" folder, and the weights of pre-trained models inside the "./weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/NIRGeneration_pages/)

### Folder Structure

```
./
├── src
└── weights
    ├── 
```

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets

Datasets are available [here] (https://zenodo.org/records/7044411) and [here] (https://zenodo.org/records/17119646)

Our architectures were evaluated on two analog effects: 
- Reverb 
- Low-pass Filter


# How To Train and Run Inference 

This code relies on Python 3.9 and TensorFlow.
First, install Python dependencies:
```
cd ./code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (default=60)
* --model - The name of the model to train ('TCN', 'LSTM', 'SSM', 'IIR', 'IR', 'TF') [str] (default=" ")
* --batch_size - The size of each batch [int] (default=1)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)
 

Example training case: 
```
cd ./code/

python starter.py --datasets reverbs --model LSTM --epochs 500
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./code/

python starter.py --datasets reverbs --model LSTM --only_inference True
```


# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```

```
