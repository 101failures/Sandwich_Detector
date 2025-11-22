# 🚀 QUICK START - BERT4ETH Run 2 Model

## What's Included
✅ All Python code for BERT4ETH sandwich detection  
✅ Pretrained BERT embeddings (753 MB)  
✅ Complete dataset: train/val/test CSV + TFRecords  
✅ Trained Run 2 model (97.03% accuracy)  
✅ Test predictions and results  

**Total: 60 files, 1.35 GB**

---

## 📂 Folder Structure
```
Final_Model_Run2/
├── 📜 Core Python Files (7 scripts)
│   ├── modeling.py              # BERT4ETH architecture
│   ├── optimization.py          # Adam optimizer
│   ├── train_with_validation.py # Training script (MAIN)
│   ├── run_sandwich_detection.py# Inference script
│   ├── gen_sandwich_data.py     # CSV → TFRecord converter
│   ├── calculate_metrics*.py    # Metrics calculation
│   └── vocab.py                 # Vocabulary utilities
│
├── 📊 Dataset (253,409 transactions)
│   ├── train.csv (177,386 - 70%)
│   ├── val.csv   (38,011 - 15%)
│   └── test.csv  (38,012 - 15%)
│
├── 🧠 bert_embeddings/ (753 MB)
│   ├── model_752000.* (checkpoint files)
│   └── vocab.txt      (address vocab)
│
├── 🗃️ tfrecords/ (79 MB)
│   ├── sandwich_train.tfrecord.*
│   ├── sandwich_val.tfrecord.*
│   └── sandwich_test.tfrecord.*
│
├── 🎯 trained_model_run2/ (450 MB)
│   ├── model_final.* (best checkpoint)
│   ├── model_step_*.* (9 checkpoints)
│   ├── detailed_training_log.json
│   ├── improved_predictions.csv
│   └── attacker_rankings_with_profit.csv
│
└── 📝 Documentation
    ├── README.md     # Detailed guide
    ├── CONTENTS.md   # Complete file inventory
    └── QUICKSTART.md # This file
```

---

## 🏃 Three Ways to Use This Model

**Note:** The repository has been reorganized into a standard Python package structure. Install the package first:

```bash
pip install -e .
```

### Option 1: Use Pretrained Model (Fastest)
```bash
# Note: Large files (pretrained models) are available by request
# Run predictions on test set using trained Run 2 model
python -m sandwich_detector.train \
  --do_predict=True \
  --test_file=./tfrecords/sandwich_test.tfrecord.sandwich_detector \
  --vocab_file=./bert_embeddings/vocab.txt \
  --init_checkpoint=./trained_model_run2/model_final \
  --output_dir=./my_predictions

# Expected: 97.03% accuracy on test set
```

### Option 2: Retrain from BERT Embeddings
```bash
# Note: Large files (BERT embeddings, TFRecords) are available by request
# Train new model using pretrained BERT embeddings
python -m sandwich_detector.train \
  --data_dir=./tfrecords \
  --vocab_file=./bert_embeddings/vocab.txt \
  --init_checkpoint=./bert_embeddings/model_752000 \
  --output_dir=./my_training \
  --batch_size=32 \
  --num_train_epochs=10 \
  --learning_rate=5e-5

# Or use the wrapper script
./scripts/train.sh --data_dir=./tfrecords --output_dir=./my_training

# Expected: ~35 minutes training, ~97% accuracy
```

### Option 3: Process New Data
```bash
# Step 1: Convert your CSV to TFRecords
python -m sandwich_detector.gen_sandwich_data \
  --data_path=your_new_data.csv \
  --output_dir=./new_tfrecords \
  --vocab_filename=./bert_embeddings/vocab.txt \
  --max_seq_length=100

# Step 2: Train on your data
python -m sandwich_detector.train \
  --data_dir=./new_tfrecords \
  --init_checkpoint=./bert_embeddings/model_752000 \
  --output_dir=./new_model \
  ...
```

---

## 📋 Requirements

### Software
```bash
# Python environment
Python >= 3.9
TensorFlow == 2.9.2 (GPU-enabled)
CUDA == 11.2
cuDNN == 8.1

# Python packages
pip install tensorflow-gpu==2.9.2
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit-learn==1.2.2
```

### Hardware
- **GPU:** NVIDIA RTX 3060/3070 or equivalent (6-8 GB VRAM)
- **RAM:** 16 GB minimum
- **Disk:** 5 GB free space
- **OS:** Windows 11, Linux, macOS (with GPU)

---

## 🎯 Model Performance (Run 2)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.03% |
| **Sandwich Detection** | 98.68% |
| **Training Time** | 35 minutes |
| **Parameters** | 4,275,836 (98.5% frozen) |

### Per-Class Results
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal | 96.90% | 98.31% | 97.60% |
| Frontrun | 98.13% | 96.97% | 97.55% |
| Victim | 97.30% | 96.97% | 97.13% |
| Backrun | 97.42% | 96.97% | 97.19% |

---

## 🔍 What Each File Does

### Core Package (`src/sandwich_detector/`)
- **`modeling.py`** - BERT4ETH model architecture (transformer, attention, classifier)
- **`optimization.py`** - Adam optimizer with warmup and weight decay
- **`train.py`** - ⭐ MAIN training script with validation monitoring
- **`gen_sandwich_data.py`** - Converts CSV to TFRecord format
- **`vocab.py`** - Address vocabulary management

### Scripts
- **`scripts/train.sh`** - Shell wrapper to run training
- **`scripts/train.py`** - Python wrapper to run training

### Tests
- **`tests/test_imports.py`** - Package import validation tests

### Key Data Files
- **CSV files** - Original transaction data (9 columns: addresses, value, gas, etc.)
- **TFRecord files** - Preprocessed binary format for TensorFlow training
- **BERT embeddings** - Pretrained address embeddings (175,493 addresses × 24-dim)
- **Trained checkpoints** - Model weights from Run 2 training

---

## ⚡ Quick Commands

```bash
# 0. Install the package
pip install -e .

# 1. Verify installation
python -c "import sandwich_detector; print('Version:', sandwich_detector.__version__)"

# 2. Check GPU availability (if TensorFlow installed)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Run training (requires large files - available by request)
python -m sandwich_detector.train \
  --data_dir=./tfrecords \
  --output_dir=./output \
  --batch_size=32

# 4. Run tests
pytest tests/ -v

# 5. View training logs (if training output exists)
cat output/training_summary.json
```

---

## 🐛 Troubleshooting

### Error: "Cannot find checkpoint"
**Solution:** Verify paths in commands match folder structure
```bash
# Correct path to trained model:
--init_checkpoint=./trained_model_run2/model_final

# Correct path to BERT embeddings:
--init_checkpoint=./bert_embeddings/model_752000
```

### Error: "Out of memory"
**Solution:** Reduce batch size
```bash
--batch_size=16  # instead of 32
```

### Error: "TensorFlow not found"
**Solution:** Install GPU-enabled TensorFlow
```bash
pip install tensorflow-gpu==2.9.2
```

### Error: "CUDA not available"
**Solution:** Verify GPU setup
```bash
# Check GPU
nvidia-smi

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

---

## 📖 Learn More

- **`README.md`** - Complete documentation with usage examples
- **`CONTENTS.md`** - Detailed file inventory with sizes and descriptions
- **`bert_config.json`** - Model hyperparameters
- **`trained_model_run2/training_summary.json`** - Run 2 training statistics

---

## 🎓 Project Info

**Institution:** Khalifa University  
**Course:** Senior Design Project 2 (SDP2)  
**Group:** CS07  
**Academic Year:** 2024-2025  
**Model:** BERT4ETH for Sandwich Attack Detection  
**Best Run:** Run 2 - 97.03% accuracy  

---

## ✅ Checklist Before Starting

- [ ] GPU with 6+ GB VRAM available
- [ ] TensorFlow 2.9.2 GPU installed
- [ ] All 60 files present (1.35 GB total)
- [ ] BERT embeddings exist (`bert_embeddings/model_752000.*`)
- [ ] TFRecords exist (`tfrecords/*.tfrecord.*`)
- [ ] Trained model exists (`trained_model_run2/model_final.*`)

---

**Ready to go! 🚀**

Start with Option 1 (inference on test set) to verify everything works, then explore training and data processing.
