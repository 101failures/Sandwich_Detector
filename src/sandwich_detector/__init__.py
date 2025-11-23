"""
Sandwich Detector - BERT4ETH-based sandwich attack detection for Ethereum

This package provides tools for detecting sandwich attacks in Ethereum transactions
using a BERT-based model architecture.
"""

__version__ = "0.1.0"

# Import main components for easy access
# Note: Some imports require TensorFlow to be installed
try:
    from sandwich_detector.modeling import BertConfig, BertModel
    from sandwich_detector.vocab import FreqVocab
    
    __all__ = [
        "BertConfig",
        "BertModel",
        "FreqVocab",
    ]
except ImportError as e:
    # TensorFlow or other dependencies not available
    # Package can still be imported but some functionality will be limited
    __all__ = []
