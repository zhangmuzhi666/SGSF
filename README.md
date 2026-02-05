
# 📂 Data & Models Preparation

We provide automated scripts to download datasets and generate the balanced 1k subsets used in our experiments.

## 1. MNIST & CIFAR-10 (Automated)

We have encapsulated the download and subset generation logic in the `util/` directory.

Please run the following commands from the **root** of this repository. These scripts will:

1. Automatically download the raw datasets to `data/torchvision/`.
2. Generate and cache the balanced 1,000-sample subsets (100 per class) to `data/cache/`.
Bash

```
# download MNIST   
python util\download_mnist.py

# download CIFAR-10 1k subset
python util/build_cifar10_1k.py
```

## 2. ImageNet & Pre-trained Models

For **ImageNet**, we use a fixed subset of 1,000 randomly selected images (pre-processed). For model checkpoints, we use standard pre-trained weights.

Please download the unified package `ALMA_models_data.zip` which contains:

- The ImageNet 1k validation subset tensor.
- Pre-trained model weights for all datasets.

## Download & Setup:

1. Download from Zenodo: https://zenodo.org/record/6549010/files/ALMA_models_data.zip
2. Unzip the file at the **root** of this repository.

# 📂 Directory Structure

```text
├── ATTACK/                 # Scripts to run attacks on different datasets
├── attacks/                # Implementation of attack algorithms (e.g., GeoSensFool)
├── data/                   # Dataset storage (raw data & cached subsets)
├── models/                 # Model definitions and checkpoints
├── Plot/                   # Scripts for plotting results and curves
└── util/                   # Utility scripts for data preparation
```

# Experiments

To run the experiments on MNIST, CIFAR10 and ImageNet, execute the scripts:

- `python minimal_attack_mnist.py`
- `python minimal_attack_cifar10.py`
- `python minimal_attack_imagenet.py`

All the results will be saved in the `results` directory as `.pt` files containing python dictionaries with information related to the attacks.

# Results

To extract all the results in a readable `.csv` file, use the `compile_results.py` script. This script contains a configuration of all the attacks run. If only a part of the experiments were performed, part of the config can be commented to account for it. This will create one `.csv` file per dataset and save them in the `results` directory.

# Curves

To plot the robust accuracy curves, the scripts `plot_results_mnist.py`, `plot_results_cifar10.py`, `plot_results_imagenet.py` can be executed. This will save the curves in the `results/curves` folder.




