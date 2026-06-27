#!/bin/bash

# Convert Python script to Jupyter notebook

echo "Converting sampige_distill_notebook.py to Jupyter notebook..."

# Check if jupytext is installed
if ! command -v jupytext &> /dev/null; then
    echo "Installing jupytext..."
    pip install jupytext -q
fi

# Convert to notebook
jupytext --to notebook sampige_distill_notebook.py --output sampige_distill.ipynb

# Check if conversion was successful
if [ -f "sampige_distill.ipynb" ]; then
    echo "✅ Successfully created sampige_distill.ipynb"
    
    # Clean up the Python file if desired
    # rm sampige_distill_notebook.py
    
    echo "Notebook is ready for Kaggle and Colab!"
else
    echo "❌ Conversion failed"
    exit 1
fi
