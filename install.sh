#!/usr/bin/bash
# Create and activate the Conda environment
conda create -n ncl python=3.10 -y
conda activate ncl

# Install required packages
pip install faiss-cpu
pip install torch==2.4.1 torchvision==0.19.1 numpy scipy scikit-learn pillow argparse tqdm

# Confirmation of installation
echo "Conda environment 'ncl' set up successfully with all required packages."