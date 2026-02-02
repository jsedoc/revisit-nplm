# Faithful NPLM Implementation Notebook

This notebook (`simple_nplm.ipynb`) provides a **complete, production-quality implementation** of the Neural Probabilistic Language Model (NPLM) directly from the repository's source code.

## Overview

The notebook implements the **full NPLM architecture** that:
- Uses adaptive token embeddings with optional projection
- Concatenates embeddings from configurable context windows
- Applies global context aggregation with learned or fixed kernels
- Processes through multi-layer NPLM decoder stack
- Supports adaptive softmax for efficient large vocabulary modeling
- Includes label smoothing and proper regularization

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
   - Load all NPLM classes from the original implementation
   - Create a synthetic dataset for demonstration
   - Build the complete NPLM model with proper configuration
   - Train the model with gradient clipping and loss tracking
   - Evaluate model performance
   - Inspect the model architecture

## What's Included

The notebook contains the **complete implementation** with:
1. **TokenEmbedding**: Adaptive embedding layer with hierarchical projections
2. **PositionEmbedding**: Sinusoidal position encoding for Transformer-N variant
3. **AdaptiveSoftmax**: Hierarchical softmax for large vocabularies
4. **NPLMFF**: Feed-forward network with configurable projections
5. **NPLMSublayer**: Residual connection wrapper with layer normalization
6. **NPLMLayer**: Core NPLM layer with context concatenation and global aggregation
7. **NPLM**: Complete model with all training capabilities
8. **Training Loop**: Production-quality training with gradient clipping
9. **Evaluation**: Perplexity and NLL calculation
10. **Synthetic Dataset**: Simple dataset for demonstration

## Fidelity to Original Implementation

This notebook contains the **actual production code** from the repository:
- ✅ Complete `TokenEmbedding` from `models/embeddings.py`
- ✅ Complete `PositionEmbedding` from `models/embeddings.py`
- ✅ Complete `AdaptiveSoftmax` from `models/adaptive_softmax.py`
- ✅ Complete `NPLM`, `NPLMLayer`, `NPLMFF`, `NPLMSublayer` from `models/nplm.py`
- ✅ Complete utility functions from `utils/__init__.py`
- ✅ All hyperparameters and architectural details preserved

**This is NOT a simplified version** - it's the actual implementation copied directly from the production source files. The only differences are:
- Uses a synthetic dataset instead of loading from disk
- Simplified training loop (no checkpointing, distributed training)
- No command-line argument parsing

## Learning Resources

- Original NPLM paper: Bengio et al. (2003) - "A Neural Probabilistic Language Model"
- Modern NPLM paper: Sun & Iyyer (2021) - "Revisiting Simple Neural Probabilistic Language Models"
- Full paper: https://arxiv.org/pdf/2104.03474.pdf

## Key Architecture Features

This implementation includes:

### Context Configuration
- **ngm** (3): Number of recent tokens concatenated with full embeddings
- **wsz** (4): Window size for distant context averaging
- **concat_layers** ([0]): Which layers apply context concatenation

### Global Aggregation
- **average**: Uniform averaging of distant context windows
- **kernel**: Learned convolution kernels for aggregation

### Model Configuration
- **embedding_size**: 256 (token embedding dimension)
- **model_size**: 256 (model hidden dimension)
- **hidden_dim**: 1024 (feed-forward expansion)
- **num_layers**: 4 (number of NPLM layers)
- **dropout_p**: 0.1 (dropout rate)

## Next Steps

After running this notebook, you can:
1. Modify `SimpleConfig` to experiment with different architectures
2. Replace `SimpleDataset` with real text data from `data/text.py`
3. Enable adaptive softmax by setting `config.adaptive = True`
4. Try Transformer-N variant by setting `config.TFN = True`
5. Train on larger datasets using the full training script with `main.py`
6. Compare with transformer baselines
