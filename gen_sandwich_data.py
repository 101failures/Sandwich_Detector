"""
Data preprocessing script for Sandwich Attack Detection
This script processes raw CSV files (train.csv, val.csv, test.csv) and converts them to TFRecord format
for training the BERT-based sandwich detection model.

Label encoding:
- 0: Non-sandwich transaction
- 1: Frontrun (first transaction in sandwich attack)
- 2: Victim (middle transaction in sandwich attack)  
- 3: Backrun (final transaction in sandwich attack)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd
import numpy as np
import collections
import random
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Model'))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from vocab import FreqVocab
import pickle as pkl
import time

flags = tf.flags
FLAGS = flags.FLAGS

random_seed = 12345
rng = random.Random(random_seed)

## parameters
flags.DEFINE_integer("max_seq_length", 100, "max sequence length.")
flags.DEFINE_string("data_dir", './Data/', "data dir.")
flags.DEFINE_string("output_dir", './inter_data/', "output dir for TFRecord files.")
flags.DEFINE_string("vocab_dir", '../../Model/inter_data/', "vocab dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_string("bizdate", "sandwich_detector", "the signature of running experiments")

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)


class SandwichInstance(object):
    """A single training instance for sandwich detection."""

    def __init__(self, from_address, to_address, transaction_features, label):
        """
        Args:
            from_address: sender address
            to_address: receiver address  
            transaction_features: dict with transaction metadata
            label: 0=non-sandwich, 1=frontrun, 2=victim, 3=backrun
        """
        self.from_address = from_address
        self.to_address = to_address
        self.txhash = transaction_features['txhash']
        self.block = transaction_features['block']
        self.transaction_index = transaction_features['transaction_index']
        self.sequence_position = transaction_features.get('sequence_position', 0)  # 0=first, 1=middle, 2=last
        self.gas = transaction_features['gas']
        self.gas_price = transaction_features['gas_price']
        self.gas_used = transaction_features['gas_used']
        self.value = transaction_features['value']
        self.block_timestamp = transaction_features['block_timestamp']
        self.nonce = transaction_features['nonce']
        self.logs_count = transaction_features['logs_count']
        self.label = label

    def __str__(self):
        s = f"from: {self.from_address}, to: {self.to_address}\n"
        s += f"label: {self.label}, block: {self.block}, tx_index: {self.transaction_index}\n"
        return s

    def __repr__(self):
        return self.__str__()


def load_csv_data(file_path):
    """Load and preprocess CSV data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df['from_address'] = df['from_address'].fillna('0x0')
    df['to_address'] = df['to_address'].fillna('0x0')
    df['gas_price'] = df['gas_price'].fillna(0)
    df['gas_used'] = df['gas_used'].fillna(0)
    df['value'] = df['value'].fillna(0)
    df['logs_count'] = df['logs_count'].fillna(0)
    df['nonce'] = df['nonce'].fillna(0)
    
    print(f"Loaded {len(df)} transactions")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    return df


def create_vocab_from_data(train_df, val_df, test_df):
    """Create vocabulary from all unique addresses in the dataset."""
    print("Creating vocabulary from addresses...")
    
    all_addresses = set()
    for df in [train_df, val_df, test_df]:
        all_addresses.update(df['from_address'].unique())
        all_addresses.update(df['to_address'].unique())
    
    # Remove NaN and null addresses
    all_addresses = {addr for addr in all_addresses if pd.notna(addr) and addr != '0x0'}
    
    print(f"Found {len(all_addresses)} unique addresses")
    
    # Create vocabulary using FreqVocab's structure
    vocab = FreqVocab()
    # Add addresses to counter with frequency 1
    for addr in all_addresses:
        vocab.counter[addr] = 1
    
    # Generate the vocabulary mapping
    vocab.generate_vocab()
    
    return vocab


def gen_sandwich_samples(df):
    """Generate sandwich detection instances from dataframe with sequence-aware features."""
    instances = []
    
    # Sort by block and transaction_index to detect sequences
    df = df.sort_values(['block', 'transaction_index']).reset_index(drop=True)
    
    print("Generating samples with sequence detection...")
    i = 0
    while i < len(df):
        row = df.iloc[i]
        label = int(row['label'])
        
        # Detect sequence position for sandwich attacks
        sequence_position = 0  # Default: standalone/first
        
        # Check if this is part of a 1-2-3 sandwich sequence
        if i < len(df) - 2:
            row2 = df.iloc[i + 1]
            row3 = df.iloc[i + 2]
            
            # If we have a 1-2-3 sequence in same block
            if (label == 1 and int(row2['label']) == 2 and int(row3['label']) == 3 and
                row['block'] == row2['block'] == row3['block']):
                # Mark positions: 0=frontrun, 1=victim, 2=backrun
                for j, seq_pos in enumerate([0, 1, 2]):
                    current_row = df.iloc[i + j]
                    transaction_features = {
                        'txhash': current_row['txhash'],
                        'block': int(current_row['block']) if pd.notna(current_row['block']) else 0,
                        'transaction_index': int(current_row['transaction_index']) if pd.notna(current_row['transaction_index']) else 0,
                        'sequence_position': seq_pos,
                        'gas': int(current_row['gas']) if pd.notna(current_row['gas']) else 0,
                        'gas_price': float(current_row['gas_price']) if pd.notna(current_row['gas_price']) else 0.0,
                        'gas_used': int(current_row['gas_used']) if pd.notna(current_row['gas_used']) else 0,
                        'value': float(current_row['value']) if pd.notna(current_row['value']) else 0.0,
                        'block_timestamp': int(current_row['block_timestamp']) if pd.notna(current_row['block_timestamp']) else 0,
                        'nonce': int(current_row['nonce']) if pd.notna(current_row['nonce']) else 0,
                        'logs_count': int(current_row['logs_count']) if pd.notna(current_row['logs_count']) else 0,
                    }
                    
                    instance = SandwichInstance(
                        from_address=str(current_row['from_address']),
                        to_address=str(current_row['to_address']),
                        transaction_features=transaction_features,
                        label=int(current_row['label'])
                    )
                    instances.append(instance)
                
                i += 3  # Skip the 3 transactions we just processed
                continue
        
        # Not part of a sequence, process normally
        transaction_features = {
            'txhash': row['txhash'],
            'block': int(row['block']) if pd.notna(row['block']) else 0,
            'transaction_index': int(row['transaction_index']) if pd.notna(row['transaction_index']) else 0,
            'sequence_position': -1,  # Standalone transaction (not in complete sandwich)
            'gas': int(row['gas']) if pd.notna(row['gas']) else 0,
            'gas_price': float(row['gas_price']) if pd.notna(row['gas_price']) else 0.0,
            'gas_used': int(row['gas_used']) if pd.notna(row['gas_used']) else 0,
            'value': float(row['value']) if pd.notna(row['value']) else 0.0,
            'block_timestamp': int(row['block_timestamp']) if pd.notna(row['block_timestamp']) else 0,
            'nonce': int(row['nonce']) if pd.notna(row['nonce']) else 0,
            'logs_count': int(row['logs_count']) if pd.notna(row['logs_count']) else 0,
        }
        
        instance = SandwichInstance(
            from_address=str(row['from_address']),
            to_address=str(row['to_address']),
            transaction_features=transaction_features,
            label=label
        )
        instances.append(instance)
        i += 1
    
    print(f"Generated {len(instances)} instances with sequence awareness")
    return instances


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def normalize_value(value, min_val=0, max_val=1e18):
    """Normalize value to [0, 1] range."""
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return float(value - min_val) / (max_val - min_val)


def write_instances_to_tfrecord(instances, vocab, output_file):
    """Write instances to TFRecord file."""
    print(f"Writing {len(instances)} instances to {output_file}...")
    
    writer = tf.python_io.TFRecordWriter(output_file)
    total_written = 0
    
    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        
        # Convert addresses to IDs
        from_addr_id = vocab.token_to_ids.get(instance.from_address, 0)
        to_addr_id = vocab.token_to_ids.get(instance.to_address, 0)
        
        # Normalize numerical features (using log scale for better distribution)
        gas_normalized = np.log1p(instance.gas) / np.log1p(1e7)
        gas_price_normalized = np.log1p(instance.gas_price) / np.log1p(1e12)
        gas_used_normalized = np.log1p(instance.gas_used) / np.log1p(1e7)
        value_normalized = np.log1p(instance.value) / np.log1p(1e19)
        
        # Normalize transaction_index to [0, 1] range (most blocks have < 500 txs)
        tx_index_normalized = min(instance.transaction_index / 500.0, 1.0)
        
        # Sequence position: -1 (standalone), 0 (frontrun), 1 (victim), 2 (backrun)
        # Normalize to [-0.33, 0, 0.33, 0.67] range to maintain clear distinction
        sequence_position_normalized = instance.sequence_position / 3.0
        
        # Create feature dictionary
        features = collections.OrderedDict()
        features["from_address_id"] = create_int_feature([from_addr_id])
        features["to_address_id"] = create_int_feature([to_addr_id])
        features["label"] = create_int_feature([instance.label])
        features["block"] = create_int_feature([instance.block])
        features["transaction_index"] = create_float_feature([tx_index_normalized])  # Now a feature!
        features["sequence_position"] = create_float_feature([sequence_position_normalized])  # New feature!
        features["gas"] = create_float_feature([gas_normalized])
        features["gas_price"] = create_float_feature([gas_price_normalized])
        features["gas_used"] = create_float_feature([gas_used_normalized])
        features["value"] = create_float_feature([value_normalized])
        features["block_timestamp"] = create_int_feature([instance.block_timestamp])
        features["nonce"] = create_int_feature([instance.nonce])
        features["logs_count"] = create_int_feature([instance.logs_count])
        
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        
        writer.write(tf_example.SerializeToString())
        total_written += 1
        
        # Print first few examples for debugging
        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info(f"from_address: {instance.from_address} (id={from_addr_id})")
            tf.logging.info(f"to_address: {instance.to_address} (id={to_addr_id})")
            tf.logging.info(f"label: {instance.label}")
            tf.logging.info(f"block: {instance.block}")
    
    writer.close()
    print(f"Wrote {total_written} total instances to {output_file}")


def main(_):
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Load data
    train_df = load_csv_data(os.path.join(FLAGS.data_dir, 'train.csv'))
    val_df = load_csv_data(os.path.join(FLAGS.data_dir, 'val.csv'))
    test_df = load_csv_data(os.path.join(FLAGS.data_dir, 'test.csv'))
    
    # Create or load vocabulary
    vocab_file = os.path.join(FLAGS.vocab_dir, f"{FLAGS.vocab_filename}.{FLAGS.bizdate}")
    
    if os.path.exists(vocab_file):
        print(f"Loading existing vocabulary from {vocab_file}")
        with open(vocab_file, "rb") as f:
            vocab = pkl.load(f)
    else:
        print(f"Creating new vocabulary...")
        vocab = create_vocab_from_data(train_df, val_df, test_df)
        print(f"Saving vocabulary to {vocab_file}")
        os.makedirs(FLAGS.vocab_dir, exist_ok=True)
        with open(vocab_file, "wb") as f:
            pkl.dump(vocab, f)
    
    print(f"Vocabulary size: {len(vocab.token_to_ids)}")
    
    # Generate instances
    train_instances = gen_sandwich_samples(train_df)
    val_instances = gen_sandwich_samples(val_df)
    test_instances = gen_sandwich_samples(test_df)
    
    # Write to TFRecord files
    train_output = os.path.join(FLAGS.output_dir, f"sandwich_train.tfrecord.{FLAGS.bizdate}")
    val_output = os.path.join(FLAGS.output_dir, f"sandwich_val.tfrecord.{FLAGS.bizdate}")
    test_output = os.path.join(FLAGS.output_dir, f"sandwich_test.tfrecord.{FLAGS.bizdate}")
    
    write_instances_to_tfrecord(train_instances, vocab, train_output)
    write_instances_to_tfrecord(val_instances, vocab, val_output)
    write_instances_to_tfrecord(test_instances, vocab, test_output)
    
    print("\n=== Data Preprocessing Complete ===")
    print(f"Train samples: {len(train_instances)}")
    print(f"Validation samples: {len(val_instances)}")
    print(f"Test samples: {len(test_instances)}")
    print(f"Vocabulary size: {len(vocab.token_to_ids)}")
    print(f"Output directory: {FLAGS.output_dir}")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
