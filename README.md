# Sandwich Attack Detection with BERT4ETH + CRF

**92.38% Sequence-Level Accuracy** in detecting MEV sandwich attacks on Ethereum blockchain

## Overview

This project uses a pre-trained BERT4ETH model with a CRF (Conditional Random Fields) layer to detect sandwich attacks in blockchain transactions. The key innovation is **preserving chronological ordering** of transactions during training, which is critical for sequence-based learning.

## Key Results

- **Sequence-Level Accuracy**: 92.38% (703/761 sandwich triplets correctly identified)
- **Token-Level Accuracy**: 99.59%
- **Zero confusion** between frontrun and backrun labels
- Validated through ablation tests: simple gas-based rule achieves only 35% accuracy, proving the model learns complex patterns

## Model Architecture

- **Base Model**: BERT4ETH (64 hidden units, 8 layers, 2 attention heads)
- **Pre-training**: 752K steps on Ethereum transaction data (checkpoint: model_752000)
- **Fine-tuning**: CRF layer with transition constraints (prevents illegal frontrun→victim transitions)
- **Features**: Gas price (10 buckets), transaction index (11 buckets), gas used (8 buckets), log count (6 buckets)

## Dataset

- **Source**: 253,409 Ethereum transactions (sandwich and normal transactions)
- **Split**: 70/15/15 (train/val/test) preserving chronological order
- **Critical Insight**: Maintaining exact blockchain ordering is essential - previous approaches grouped by address, destroying the temporal patterns that sandwich attacks exhibit

## Project Structure

```
SANDWICH_DETECTION_FINAL/
├── model/                      # Model architecture
│   ├── bert_config.json       # BERT configuration
│   ├── modeling.py            # BERT4ETH implementation
│   ├── crf_layer.py           # CRF layer with constraints
│   ├── optimization.py        # Training optimizer
│   └── vocab.py               # Feature vocabulary
├── scripts/                    # Training & evaluation
│   ├── train_crf.py           # Training script
│   ├── evaluate_crf.py        # Evaluation script
│   ├── prepare_data_blockchain_order.py  # Data preparation
│   └── split_preserve_order.py           # Dataset splitting
├── results/
│   └── confusion_matrix.png   # Visualization
├── DEMO_FOR_INSTRUCTOR.py     # Interactive demo
└── RESULTS_REPORT.md          # Detailed analysis
```

## Quick Start

### Requirements

```bash
pip install tensorflow==2.9.2 tensorflow-addons==0.22.0 pandas numpy scikit-learn matplotlib
```

### Training

1. Prepare data (preserving chronological order):
```bash
python scripts/prepare_data_blockchain_order.py
```

2. Train model:
```bash
python scripts/train_crf.py
```

3. Evaluate:
```bash
python scripts/evaluate_crf.py
```

### Demo

Run the interactive demonstration:
```bash
python DEMO_FOR_INSTRUCTOR.py
```

## Validation

The model has been validated to ensure it learns real patterns, not simple heuristics:

1. **Ablation Test**: A simple gas-based rule achieves only 35% accuracy, while the model achieves 99.59%
2. **Confusion Matrix**: Zero confusion between frontrun/backrun classes
3. **Sequence Recovery**: 92.38% of complete sandwich triplets correctly identified

See `RESULTS_REPORT.md` for comprehensive analysis.

## Key Breakthrough

The critical breakthrough was **preserving chronological ordering** during dataset preparation. Previous approaches grouped transactions by address, which destroyed the temporal patterns inherent in sandwich attacks (frontrun → victim → backrun sequence). By maintaining exact blockchain order, the model successfully learns these sequential dependencies.

## Hardware

- **GPU**: NVIDIA RTX 3080 (8GB) or equivalent
- **Training Time**: ~6 hours
- **Memory**: 8GB GPU RAM, 16GB System RAM

## Citation

This project builds on BERT4ETH:

```bibtex
@inproceedings{hu2023bert4eth,
  title={Blockchain Transaction Representation Learning with Transformer},
  author={Hu, Di and others},
  booktitle={WWW 2023},
  year={2023}
}
```

## License

Educational and research purposes.

