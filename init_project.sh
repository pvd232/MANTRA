#!/bin/bash

# Project Initialization Script
# Usage: source init_project.sh

echo "Initializing MANTRA environment..."

# 1. Activate Conda Environment
# (Adjust 'mantra_env' to match the name in configs/env.yml)
ENV_NAME=$(grep "name:" configs/env.yml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    echo "Error: Could not find environment name in configs/env.yml"
    return 1
fi

echo "Activating conda environment: $ENV_NAME"
# Try to find conda profile if not loaded
if [ -z "$CONDA_PREFIX" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate $ENV_NAME

# 2. Add Project Root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Environment ready! 🚀"
