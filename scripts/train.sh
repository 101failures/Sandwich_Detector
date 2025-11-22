#!/bin/bash
# Wrapper script to run sandwich detector training
# Usage: ./scripts/train.sh [args...]
# Example: ./scripts/train.sh --output_dir=./output --data_dir=./tfrecords

python -m sandwich_detector.train "$@"
