#!/bin/bash

set -e

# MSD: http://medicaldecathlon.com
mkdir -p msd
# uvx gdown "https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view" --fuzzy -O msd/Task01_BrainTumour.tar
uvx gdown "https://drive.google.com/file/d/1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY/view" --fuzzy -O msd/Task02_Heart.tar
# uvx gdown "https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view" --fuzzy -O msd/Task03_Liver.tar  # too big: 27GB
uvx gdown "https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view" --fuzzy -O msd/Task04_Hippocampus.tar
uvx gdown "https://drive.google.com/file/d/1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a/view" --fuzzy -O msd/Task05_Prostate.tar
# uvx gdown "https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view" --fuzzy -O msd/Task07_Pancreas.tar
# uvx gdown "https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view" --fuzzy -O msd/Task08_HepaticVessel.tar
uvx gdown "https://drive.google.com/file/d/1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE/view" --fuzzy -O msd/Task09_Spleen.tar
# uvx gdown "https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view" --fuzzy -O msd/Task10_Colon.tar
# extract
tar -xvf msd/Task02_Heart.tar -C msd
tar -xvf msd/Task04_Hippocampus.tar -C msd
tar -xvf msd/Task05_Prostate.tar -C msd
tar -xvf msd/Task09_Spleen.tar -C msd
rm msd/Task*/.*
rm msd/Task*/*/.*

### MedSAM
# FLARE22: https://zenodo.org/records/7860267
mkdir -p flare22
wget "https://zenodo.org/records/7860267/files/FLARE22Train.zip?download=1" -O flare22.zip
unzip flare22.zip -d flare22
rm flare22.zip
# Model weights
mkdir -p medsam
uvx gdown "https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view" --fuzzy -O medsam/medsam_vit_b.pth

# Kather100k: https://zenodo.org/records/1214456
mkdir -p kather100k
wget "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1" -O kather100k.zip
unzip kather100k.zip -d kather100k
rm kather100k.zip

# FlowMap
uvx gdown "https://drive.google.com/file/d/1aaru7A8cDfyZ_d0bgErpYGh5uksHljoN/view" --fuzzy -O flowmap_datasets.zip
unzip flowmap_datasets.zip -d flowmap_datasets
rm flowmap_datasets.zip

# TGCA-CRC for PathFinder
mkdir -p pathfinder
wget "https://cloud.tsinghua.edu.cn/f/a1c87bb480eb4eae9bd2/?dl=1" -O pathfinder.zip
unzip pathfinder.zip -d pathfinder
rm pathfinder.zip

# CPTAC for PathFinder
uvx gdown --fuzzy -O pathfinder_cptac.tar.gz "https://drive.google.com/file/d/1shCsgWP4ctbcMoZ3CVelfh49AXa0RcJT/view"
mkdir -p pathfinder_cptac
tar -xvf pathfinder_cptac.tar.gz -C pathfinder_cptac
mv pathfinder_cptac/prob_map pathfinder/CPTAC_CRC
rmdir pathfinder_cptac
rm pathfinder_cptac.tar.gz

# scGPT
mkdir -p scgpt
wget "https://figshare.com/ndownloader/files/24539828" -O scgpt/human_pancreas_norm_complexBatch.h5ad

# cucumber
wget "https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg" -O cucumber.jpg

# TabPFN
mkdir -p tabpfn
uv run --with scikit-learn python tabpfn.py