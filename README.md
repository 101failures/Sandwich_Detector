# BERT4ETH-based Sandwich Attack Detection

This folder contains all the code, model weights, embeddings, and data used for the final BERT4ETH-based sandwich attack detection model (Run 2 - 97.03% accuracy).
#### Note: large files (like the BERT checkpoint and our training dataset) are available upon request, as they are too large to be uploaded in this repository.
## Contents

### Core Model Files

#### Data Processing
- `gen_sandwich_data.py` - Generates TFRecord files from raw CSV data
- `vocab.py` - Vocabulary management for address embeddings

#### Model Architecture
- `modeling.py` - BERT4ETH model architecture implementation
- `optimization.py` - Adam optimizer with warmup and weight decay
- `bert_config.json` - Model configuration (768 hidden size, 12 layers, 12 heads)

#### Training & Evaluation
- `train_with_validation.py` - Main training script with validation
- `run_sandwich_detection.py` - Original training/inference script
- `calculate_metrics_with_validation.py` - Metrics calculation script

### Pretrained Embeddings
- `bert_embeddings/` - Pretrained BERT4ETH address embeddings
  - `model_752000.*` - Checkpoint files with 175,493 Ethereum address embeddings (24-dim)
  - `vocab.txt` - Address vocabulary file
  - 4,211,832 frozen parameters (98.5% of total model)

### Dataset
- `train.csv` - Training set (177,386 transactions, 70%)
- `val.csv` - Validation set (38,011 transactions, 15%)
- `test.csv` - Test set (38,012 transactions, 15%)
- Columns: from, to, value, gas, gasPrice, timestamp, logs_count, transaction_index, sequence_position, label
- Labels: 0=Normal, 1=Frontrun, 2=Victim, 3=Backrun

### TFRecord Files
- `tfrecords/sandwich_train.tfrecord.sandwich_detector` - Preprocessed training data
- `tfrecords/sandwich_val.tfrecord.sandwich_detector` - Preprocessed validation data
- `tfrecords/sandwich_test.tfrecord.sandwich_detector` - Preprocessed test data

### Trained Model
- `trained_model_run2/` - Best performing model checkpoint
  - Model weights achieving 97.03% test accuracy
  - Checkpoint files for inference
  - Training logs and validation metrics

### Test Results
- `test_predictions/` - Test set predictions
  - Predictions CSV with all 38,012 test samples
  - True labels and predicted probabilities

## Installation

```bash
# Install the package in development mode
pip install -e .
```

## How to Use

### 1. Using as a Python Package

```python
# Import the package components
from sandwich_detector import BertConfig, BertModel, FreqVocab

# Use in your code
config = BertConfig(vocab_size=175493)
```

### 2. Data Preparation

```bash
# Generate TFRecord files from CSV
python -m sandwich_detector.gen_sandwich_data \
  --data_path=sandwich_dataset_final.csv \
  --output_dir=./tfrecords \
  --vocab_filename=./bert_embeddings/vocab.txt \
  --max_seq_length=100
```

### 3. Training

```bash
# Train BERT4ETH model using the module
python -m sandwich_detector.train \
  --train_file=./tfrecords/train.tfrecord \
  --val_file=./tfrecords/val.tfrecord \
  --test_file=./tfrecords/test.tfrecord \
  --vocab_filename=./bert_embeddings/vocab.txt \
  --init_checkpoint=./bert_embeddings/model.ckpt \
  --output_dir=./output \
  --num_train_epochs=10 \
  --batch_size=32 \
  --learning_rate=5e-5

# Or use the wrapper script
./scripts/train.sh --output_dir=./output --data_dir=./tfrecords
```

**Note:** Large files (pretrained embeddings, TFRecord files, trained models) are not included in this repository due to size constraints. These files are available by request.

## Model Architecture

**Input Layer:**
- Transaction features: from_address, to_address, value, gas, gasPrice, logs_count, transaction_index, sequence_position

**Embedding Layer:**
- Address embeddings: 175,493 addresses × 24 dims (pretrained, frozen)
- Feature embeddings: 6 numerical features (trainable)
- Concatenated: 54-dimensional input

**BERT Encoder:**
- 12 transformer layers
- 768 hidden size
- 12 attention heads
- Multi-head self-attention + feed-forward networks

**Classification Head:**
- Dense(768 → 256) + ReLU + Dropout(0.3)
- Dense(256 → 128) + ReLU + Dropout(0.3)
- Dense(128 → 128) + ReLU + Dropout(0.3)
- Output(128 → 4) + Softmax

**Total Parameters:** 4,275,836
- Frozen (embeddings): 4,211,832 (98.5%)
- Trainable (classifier): 63,236 (1.5%)

## Training Configuration

**Hyperparameters:**
- Batch size: 32
- Learning rate: 5e-5
- Epochs: 10
- Optimizer: Adam
- Warmup steps: 100
- Weight decay: 0.01
- Dropout rate: 0.3

**Dataset Split:**
- Training: 177,386 (70%)
- Validation: 38,011 (15%)
- Test: 38,012 (15%)

## Performance

**Overall Metrics:**
- Test Accuracy: 97.03%
- Sandwich Detection Rate: 98.68%
- Training Time: 35 minutes (GPU)

**Per-Class Performance:**
- Normal: Precision 0.9690, Recall 0.9831, F1 0.9760
- Frontrun: Precision 0.9813, Recall 0.9697, F1 0.9755
- Victim: Precision 0.9730, Recall 0.9697, F1 0.9713
- Backrun: Precision 0.9742, Recall 0.9697, F1 0.9719

## Requirements

```bash
Python >= 3.9
TensorFlow >= 2.9.2 (GPU-enabled recommended)
numpy >= 1.23.5
pandas >= 1.5.3
scikit-learn >= 1.2.2
```

## Development

To contribute or run tests locally:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
flake8 src/sandwich_detector --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Hardware Requirements

- GPU: NVIDIA RTX 3060/3070 or equivalent (6-8 GB VRAM)
- RAM: 16 GB minimum
- Storage: 5 GB for model and data

## Key Features

1. **Transfer Learning**: Leverages pretrained Ethereum address embeddings
2. **Frozen Embeddings**: 98.5% of parameters frozen for efficient training
3. **Attention Mechanism**: Captures sequential patterns in sandwich attacks
4. **Class Balancing**: Handles 10:1 class imbalance with stratified sampling
5. **Fast Training**: 35 minutes on GPU vs 42 minutes for LSTM baseline

## Citation

If you use this model, please cite:

```
BERT4ETH Sandwich Attack Detection
Khalifa University - Senior Design Project
Group CS07, Academic Year 2024-2025
```

## Repository

Original BERT4ETH: https://github.com/git-disl/BERT4ETH

## Notes

- The pretrained embeddings (`bert_embeddings/`) are essential for model performance
- TFRecord format is required for efficient training with TensorFlow
- Model checkpoint files (.ckpt) contain the trained weights
- Random seed fixed at 42 for reproducibility

