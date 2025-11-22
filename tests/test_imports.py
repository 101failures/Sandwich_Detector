"""
Test basic package imports and structure
"""
import sys
import pytest


def _tensorflow_available():
    """Check if TensorFlow is available"""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


def test_package_import():
    """Test that the package can be imported"""
    import sandwich_detector
    assert sandwich_detector.__version__ == "0.1.0"


def test_vocab_import():
    """Test that vocab module can be imported"""
    from sandwich_detector import vocab
    assert hasattr(vocab, 'FreqVocab')


@pytest.mark.skipif(
    not _tensorflow_available(),
    reason="TensorFlow not available"
)
def test_modeling_import():
    """Test that modeling module can be imported (requires TensorFlow)"""
    from sandwich_detector import modeling
    assert hasattr(modeling, 'BertConfig')
    assert hasattr(modeling, 'BertModel')


@pytest.mark.skipif(
    not _tensorflow_available(),
    reason="TensorFlow not available"
)
def test_optimization_import():
    """Test that optimization module can be imported (requires TensorFlow)"""
    from sandwich_detector import optimization
    assert hasattr(optimization, 'create_optimizer')


@pytest.mark.skipif(
    not _tensorflow_available(),
    reason="TensorFlow not available"
)
def test_main_exports():
    """Test that main classes are exported in __init__ (requires TensorFlow)"""
    from sandwich_detector import BertConfig, BertModel, FreqVocab
    assert BertConfig is not None
    assert BertModel is not None
    assert FreqVocab is not None


@pytest.mark.skipif(
    not _tensorflow_available(),
    reason="TensorFlow not available"
)
def test_train_module_exists():
    """Test that train module exists and can be imported (requires TensorFlow)"""
    from sandwich_detector import train
    assert train is not None


@pytest.mark.skipif(
    not _tensorflow_available(),
    reason="TensorFlow not available"
)
def test_gen_sandwich_data_module_exists():
    """Test that gen_sandwich_data module exists and can be imported (requires TensorFlow)"""
    from sandwich_detector import gen_sandwich_data
    assert gen_sandwich_data is not None
