#!/bin/bash

# Set environment name (change if needed)
ENV_NAME="datacompos"

# Create the environment from the YAML file
echo "Creating conda environment: $ENV_NAME"
conda env create -f environment.yaml -n $ENV_NAME

echo "Done. Activate with: conda activate $ENV_NAME"
