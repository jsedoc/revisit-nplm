# Simple NPLM Notebook

This notebook (`simple_nplm.ipynb`) provides a minimal, self-contained implementation of the Neural Probabilistic Language Model (NPLM).

## Overview

The notebook implements a simplified version of the NPLM architecture that:
- Uses word embeddings to represent tokens
- Concatenates embeddings from a fixed context window
- Processes them through a feed-forward neural network
- Predicts the next word in a sequence

## Requirements

Install the required packages:

```bash
pip install -r simple_nplm_requirements.txt
```

Or manually install:
- `torch>=1.7.0`
- `numpy>=1.19.0`
- `matplotlib>=3.1.0`
- `scikit-learn>=0.24.0`

## Running the Notebook

1. Open the notebook in Jupyter:
```bash
jupyter notebook simple_nplm.ipynb
```

2. Or use JupyterLab:
```bash
jupyter lab simple_nplm.ipynb
```

3. Run all cells sequentially to:
   - Create a simple text dataset
   - Build and train the NPLM model
   - Evaluate model performance
   - Generate text
   - Visualize word embeddings

## What's Included

The notebook contains:
1. **Data Preprocessing**: Simple tokenization and vocabulary building
2. **Model Architecture**: Clean PyTorch implementation of NPLM
3. **Training Loop**: Basic training with loss tracking
4. **Evaluation**: Perplexity calculation and next-word prediction
5. **Text Generation**: Autoregressive text generation
6. **Visualization**: Training loss curves and word embedding plots

## Key Differences from Full Implementation

This is a simplified educational implementation. The full NPLM model in this repository includes:
- Adaptive softmax for large vocabularies
- Distant context aggregation with learned kernels
- Multiple decoder layers
- Transformer hybrid architectures (Transformer-N)
- Distributed training support

For production use cases, refer to the full implementation in the `models/` directory.

## Learning Resources

- Original NPLM paper: Bengio et al. (2003) - "A Neural Probabilistic Language Model"
- Modern NPLM paper: Sun & Iyyer (2021) - "Revisiting Simple Neural Probabilistic Language Models"
- Full paper: https://arxiv.org/pdf/2104.03474.pdf

## Next Steps

After understanding this simple implementation, you can:
1. Explore the full NPLM implementation in `models/nplm.py`
2. Train on larger datasets using the main training script
3. Experiment with different architectures (Transformer, Transformer-N)
4. Compare perplexity scores on benchmark datasets
