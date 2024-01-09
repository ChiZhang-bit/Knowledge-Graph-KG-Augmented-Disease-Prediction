# KEIM: Knowledge Graph Empowered Interpretable Model for Diagnosis Prediction

This repository contains our code for the DASFAA 2024 paper: **KEIM: Knowledge Graph Empowered Interpretable Model for Diagnosis Prediction**

![1](docs\1.png)

## Abstract

Electronic Health Records (EHR) include various sources of healthcare data collected from patients in hospitals. 
These data are typically stored in structured formats and are widely used in various big data healthcare analysis applications, particularly diagnosis prediction. 
Deep learning methods have achieved record-breaking results in various real-world prediction tasks. 
However, deep learning methods usually require a large amount of data for training, and the medical features that rarely appear in the data also pose great challenges for deep learning models.
Besides, while deep learning models often achieve high accuracy, the lack of interpretation remains a problem for healthcare applications, which are naturally high-stakes. 
Existing works utilize medical ontology knowledge to enhance prediction performance and provide interpretable prediction results. 
Nevertheless, the ontology knowledge is coarse-grained, where many medical concepts and relationships are not included. 
In this paper, we propose to incorporate large-scale medical knowledge graphs (KGs) into our designed model, called **KEIM**: （**K**nowledge graph **E**mpowered **I**nterpretable **M**odel), for diagnosis prediction. 
Specifically, the KGs are first integrated into the time-series module of the model via a laplacian regularization to take advantage of the complex relationships among medical features. 
Subsequently, we construct a personalized KG for each visit and design a relation-aware attentive graph neural network based on the personalized KG to augment the time-series module for interpretable predictions. 
Extensive experiments on two benchmark healthcare datasets, namely, MIMIC-III and MIMIC-IV, show that our proposed KEIM not only achieves significant improvement in terms of AUC but also provides interpretability for diagnosis prediction with KGs.

## Usage

This codebase is tested with `python=3.8`, `torch==1.11.0` and `CUDA 11.3`

### Requirements

```
scikit-learn==0.24.1
pandas==1.2.4
numpy==1.20.1
scipy
six
tqdm
```

### Data

The dataset used in our code is sourced from MIMIC III and MIMIC IV. Our processed data will be uploaded to Google Cloud later.

The data file needs to be placed in the root directory.

Your folder structure should end up looking like this:

```
└── Knowledge-Graph-KG-Augmented-Disease-Prediction  
    └── docs
    └── data
    └── saved_model
    └── src
        └── baseline
        └── model
        ├── Dataset.py
        ├── train.py
        ├── utils.py
```

### Train and Evaluation

Run `train.py`. The corresponding parameters are also in the `train.py` file.

```
usage: train.py [-h] [--model {Dip_l,Dip_g,Dip_c,Retain,LSTM}]
                [--input_dim INPUT_DIM] [--hidden_dim HIDDEN_DIM]
                [--output_dim OUTPUT_DIM] [--bi_direction]
                [--batch_size BATCH_SIZE] [--decay DECAY] [--beta BETA]
                [--p P] [--lr LR] [--only_dipole]
```

## Contact

For help and issues associated with Annotator, or reporting a bug, please open a Github issues, or feel free to contact zhangchi5675@163.com