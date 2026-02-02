# NPLM Notebook - Faithful Implementation

This notebook (`simple_nplm.ipynb`) provides a **faithful, production-quality implementation** of the Neural Probabilistic Language Model (NPLM) as implemented in this repository.

> **Note**: The notebook is named "simple" for consistency with the original request, but the implementation is **not simplified** - it contains the complete, faithful production code from the repository.

## Overview

The notebook implements the **complete NPLM architecture** by cherry-picking code directly from the source files:
- Full multi-layer NPLM decoder with context concatenation
- Adaptive softmax for efficient large vocabulary handling
- Global context aggregation with learned kernels
- Token and position embeddings with hierarchical projections
- Complete training and evaluation loops
- All original hyperparameters and architectural details

## Requirements

The notebook requires the same dependencies as the main repository. Install from the repository's requirements.txt:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=1.7.1`
- `numpy>=1.19.5`
- `tqdm>=4.32.1`

## Running the Notebook

1. Open the notebook in Jupyter:
```bash
jupyter notebook simple_nplm.ipynb
```

2. Run all cells sequentially to:
   - Import all necessary modules and classes
   - Define the complete NPLM architecture
   - Create a synthetic dataset
   - Train the NPLM model
   - Evaluate model performance (perplexity)

## What's Included

The notebook contains the complete production code:

### Core Model Components
1. **TokenEmbedding** - Adaptive embeddings with hierarchical projections (from `models/embeddings.py`)
2. **AdaptiveSoftmax** - Hierarchical softmax for large vocabularies (from `models/adaptive_softmax.py`)
3. **NPLMSublayer** - Residual connection wrapper with LayerNorm (from `models/nplm.py`)
4. **NPLMFF** - Feed-forward network with configurable projections (from `models/nplm.py`)
5. **NPLMLayer** - Core NPLM decoder with context concatenation and global aggregation (from `models/nplm.py`)
6. **NPLM** - Complete multi-layer model architecture (from `models/nplm.py`)

### Training & Evaluation
7. **Training Loop** - Full training procedure with gradient clipping (based on `actions/train.py`)
8. **Evaluation** - Perplexity calculation on validation data (based on `actions/evaluate.py`)

### Utilities
9. **Helper Functions** - All utility functions from `utils/__init__.py`

## Fidelity to Original Implementation

**This is NOT a simplified version.** The notebook contains:

✅ Complete production code copied directly from source files  
✅ All classes from `models/nplm.py`, `models/embeddings.py`, `models/adaptive_softmax.py`  
✅ Full architectural complexity preserved (no simplifications)  
✅ Original hyperparameters and initialization schemes  
✅ Context configuration with local and distant context aggregation  
✅ Adaptive softmax with hierarchical projections  
✅ Weight tying and projection tying support  

### Differences from Main Training Script

The notebook differs from `main.py` only in:
- **Dataset**: Uses a synthetic dataset created in the notebook
- **Single-GPU**: No DataParallel wrapper
- **Fewer steps**: Trains for demonstration purposes

**For distributed training** on **large datasets**, use the main training script (`main.py`) instead.

## Learning Resources

- **Original NPLM paper**: Bengio et al. (2003)
- **Modern NPLM paper**: Sun & Iyyer (2021) - https://arxiv.org/pdf/2104.03474.pdf
- **Full implementation**: See `models/nplm.py`, `main.py`, `actions/train.py`
