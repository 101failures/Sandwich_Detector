"""
Convert sandwich CSV to TFRecord - PRESERVE CSV ORDER
CSV is already organized with sandwich triplets grouped together.
Simply read sequentially and create fixed-size windows.
"""

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle as pkl
from tqdm import tqdm
import sys
sys.path.append("../Model")
from vocab import FreqVocab

MAX_SEQ_LENGTH = 100

def value_to_bucket(value):
    """Bucket transaction values."""
    if value == 0:
        return 0
    elif value < 1e15:  # < 0.001 ETH
        return 1
    elif value < 1e16:  # < 0.01 ETH
        return 2
    elif value < 1e17:  # < 0.1 ETH
        return 3
    elif value < 1e18:  # < 1 ETH
        return 4
    elif value < 1e19:  # < 10 ETH
        return 5
    elif value < 1e20:  # < 100 ETH
        return 6
    else:
        return 7

def gas_price_to_bucket(gas_price):
    """Bucket gas prices - CRITICAL for frontrun detection."""
    if pd.isna(gas_price) or gas_price == 0:
        return 0
    elif gas_price < 1e9:  # < 1 Gwei
        return 1
    elif gas_price < 5e9:  # < 5 Gwei
        return 2
    elif gas_price < 10e9:  # < 10 Gwei
        return 3
    elif gas_price < 20e9:  # < 20 Gwei
        return 4
    elif gas_price < 50e9:  # < 50 Gwei
        return 5
    elif gas_price < 100e9:  # < 100 Gwei
        return 6
    elif gas_price < 200e9:  # < 200 Gwei
        return 7
    elif gas_price < 500e9:  # < 500 Gwei
        return 8
    else:
        return 9

def gas_used_to_bucket(gas_used):
    """Bucket gas used - victim typically uses more gas."""
    if pd.isna(gas_used) or gas_used == 0:
        return 0
    elif gas_used < 30000:  # Simple transfer
        return 1
    elif gas_used < 50000:
        return 2
    elif gas_used < 100000:
        return 3
    elif gas_used < 150000:
        return 4
    elif gas_used < 250000:  # Complex swap
        return 5
    elif gas_used < 500000:
        return 6
    else:
        return 7

def logs_count_to_bucket(logs_count):
    """Bucket number of logs - victim emits more events."""
    if pd.isna(logs_count):
        return 0
    logs_count = int(logs_count)
    if logs_count == 0:
        return 0
    elif logs_count <= 2:
        return 1
    elif logs_count <= 5:
        return 2
    elif logs_count <= 10:
        return 3
    elif logs_count <= 20:
        return 4
    else:
        return 5

def priority_fee_to_bucket(fee):
    """Bucket priority fees."""
    if pd.isna(fee) or fee == 0:
        return 0
    elif fee < 1e9:
        return 1
    elif fee < 5e9:
        return 2
    elif fee < 10e9:
        return 3
    else:
        return 4

def nonce_to_bucket(nonce):
    """Bucket nonces."""
    if pd.isna(nonce):
        return 0
    if nonce < 10:
        return 1
    elif nonce < 100:
        return 2
    elif nonce < 1000:
        return 3
    else:
        return 4

def tx_type_to_bucket(tx_type):
    """Bucket transaction types."""
    if pd.isna(tx_type):
        return 0
    return min(int(tx_type), 3)

def create_sequence(txs, vocab, start_idx):
    """
    Create a sequence of transactions in BLOCKCHAIN ORDER.
    Takes up to MAX_SEQ_LENGTH transactions starting from start_idx.
    """
    # CLS token
    input_ids = [0]
    positions = [0]
    io_flags = [0]
    values = [0]
    gas_prices = [0]
    tx_indices = [0]
    gas_useds = [0]
    priority_fees = [0]
    logs_counts = [0]
    nonces = [0]
    tx_types = [0]
    counts = [0]
    labels = [0]
    
    # Take up to MAX_SEQ_LENGTH-1 transactions (leave room for CLS)
    end_idx = min(start_idx + MAX_SEQ_LENGTH - 1, len(txs))
    sequence_txs = txs[start_idx:end_idx]
    
    # Calculate relative tx_index WITHIN THIS WINDOW
    seq_tx_indices = [tx['transaction_index'] for tx in sequence_txs]
    min_tx_idx = min(seq_tx_indices) if seq_tx_indices else 0
    max_tx_idx = max(seq_tx_indices) if seq_tx_indices else 0
    
    last_timestamp = None
    position_idx = 0
    
    for tx in sequence_txs:
        # Token ID
        to_addr = tx['to_address']
        token_id = vocab.token_to_ids.get(to_addr, 1)  # 1 = [UNK]
        input_ids.append(token_id)
        
        # Position (based on timestamp changes)
        if last_timestamp is None or tx['timestamp'] != last_timestamp:
            position_idx += 1
            last_timestamp = tx['timestamp']
        positions.append(position_idx)
        
        # IO flag
        io_flags.append(1)
        
        # Features
        values.append(value_to_bucket(tx['value']))
        gas_prices.append(gas_price_to_bucket(tx['gas_price']))
        gas_useds.append(gas_used_to_bucket(tx['gas_used']))
        priority_fees.append(priority_fee_to_bucket(tx['max_priority_fee_per_gas']))
        logs_counts.append(logs_count_to_bucket(tx['logs_count']))
        nonces.append(nonce_to_bucket(tx['nonce']))
        tx_types.append(tx_type_to_bucket(tx['transaction_type']))
        
        # Relative tx_index within this sequence window
        if max_tx_idx > min_tx_idx:
            relative_pos = (tx['transaction_index'] - min_tx_idx) / (max_tx_idx - min_tx_idx)
            tx_index_bucket = min(10, max(1, int(relative_pos * 9) + 1))
        else:
            tx_index_bucket = 5
        tx_indices.append(tx_index_bucket)
        
        # Count and label
        counts.append(1)
        labels.append(tx['label'])
    
    # Pad to max length
    seq_len = len(input_ids)
    mask = [1] * seq_len + [0] * (MAX_SEQ_LENGTH - seq_len)
    
    for feature_list in [input_ids, positions, io_flags, values, gas_prices, 
                         tx_indices, gas_useds, priority_fees, logs_counts, 
                         nonces, tx_types, counts, labels]:
        feature_list.extend([0] * (MAX_SEQ_LENGTH - len(feature_list)))
    
    return {
        'input_ids': input_ids,
        'positions': positions,
        'io_flags': io_flags,
        'values': values,
        'gas_prices': gas_prices,
        'tx_indices': tx_indices,
        'gas_useds': gas_useds,
        'priority_fees': priority_fees,
        'logs_counts': logs_counts,
        'nonces': nonces,
        'tx_types': tx_types,
        'counts': counts,
        'labels': labels,
        'mask': mask
    }

def process_csv_to_tfrecord(csv_path, tfrecord_path, vocab):
    """Process CSV to TFRecord maintaining blockchain chronological order."""
    print(f"\nProcessing {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total transactions: {len(df):,}")
    
    # Check label distribution
    label_dist = df['label'].value_counts().sort_index()
    print("\nLabel distribution:")
    for label, count in label_dist.items():
        label_name = ['non-sandwich', 'frontrun', 'victim', 'backrun'][label]
        print(f"  {label} ({label_name}): {count:,} ({count/len(df)*100:.1f}%)")
    
    # CSV is already in blockchain chronological order - DO NOT RESORT
    # Just convert to list of transactions preserving exact order
    txs = []
    for _, row in df.iterrows():
        txs.append({
            'to_address': row['to_address'] if pd.notna(row['to_address']) else 'UNKNOWN',
            'block': row['block'],
            'timestamp': row['block_timestamp'],
            'value': row['value'],
            'gas': row['gas'],
            'gas_price': row['gas_price'],
            'gas_used': row['gas_used'],
            'transaction_index': row['transaction_index'],
            'max_priority_fee_per_gas': row['max_priority_fee_per_gas'] if pd.notna(row['max_priority_fee_per_gas']) else 0,
            'logs_count': row['logs_count'],
            'nonce': row['nonce'],
            'transaction_type': row['transaction_type'],
            'label': row['label'],
        })
    
    # Create sliding windows over chronological transactions
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    num_sequences = 0
    
    print(f"\nCreating sequences (sliding window over {len(txs)} transactions in original CSV order)...")
    
    # Create overlapping windows to ensure we capture all sandwiches
    stride = MAX_SEQ_LENGTH // 2  # 50% overlap to catch sandwich triplets
    
    for start_idx in tqdm(range(0, len(txs), stride)):
        if start_idx + 3 > len(txs):  # Need at least 3 txs
            break
        
        seq = create_sequence(txs, vocab, start_idx)
        
        # Write to TFRecord
        features = {
            'address': _int64_feature([0]),  # Dummy address (not used anymore)
            'input_ids': _int64_feature(seq['input_ids']),
            'input_positions': _int64_feature(seq['positions']),
            'input_io_flags': _int64_feature(seq['io_flags']),
            'input_values': _int64_feature(seq['values']),
            'input_gas_prices': _int64_feature(seq['gas_prices']),
            'input_tx_indices': _int64_feature(seq['tx_indices']),
            'input_gas_useds': _int64_feature(seq['gas_useds']),
            'input_priority_fees': _int64_feature(seq['priority_fees']),
            'input_logs_counts': _int64_feature(seq['logs_counts']),
            'input_nonces': _int64_feature(seq['nonces']),
            'input_tx_types': _int64_feature(seq['tx_types']),
            'input_counts': _int64_feature(seq['counts']),
            'input_mask': _int64_feature(seq['mask']),
            'label_ids': _int64_feature(seq['labels']),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
        num_sequences += 1
    
    writer.close()
    print(f"✓ Created {num_sequences:,} sequences in {tfrecord_path}")
    return num_sequences

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

class SimpleVocab:
    """Simple vocabulary for address tokens."""
    def __init__(self):
        self.token_to_ids = {}
        self.ids_to_token = {}

def main():
    print("="*80)
    print("CSV to TFRecord Conversion - PRESERVE EXACT CSV ORDER")
    print("="*80)
    print("\nCSV is pre-organized with sandwich triplets grouped together.")
    print("Reading sequentially without any reordering.\n")
    
    # Build vocab from all CSVs
    print("\nBuilding vocabulary...")
    all_addresses = set()
    
    for csv_file in ['./data/train.csv', 
                     './data/val.csv', 
                     './data/test.csv']:
        df = pd.read_csv(csv_file)
        all_addresses.update(df['to_address'].dropna().unique())
    
    print(f"Found {len(all_addresses):,} unique addresses")
    
    # Create vocab
    vocab = SimpleVocab()
    vocab.token_to_ids['[PAD]'] = 0
    vocab.token_to_ids['[UNK]'] = 1
    
    for i, addr in enumerate(sorted(all_addresses), start=2):
        vocab.token_to_ids[addr] = i
    
    vocab.ids_to_token = {v: k for k, v in vocab.token_to_ids.items()}
    
    print(f"Vocabulary size: {len(vocab.token_to_ids):,}")
    
    # Save vocab
    os.makedirs('./data', exist_ok=True)
    with open('./data/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)
    print("✓ Saved vocabulary to ./data/vocab.pkl")
    
    # Process each split
    total_sequences = 0
    for split, csv_path, tfr_path in [
        ('train', './data/train.csv', './data/train.tfrecord'),
        ('val', './data/val.csv', './data/val.tfrecord'),
        ('test', './data/test.csv', './data/test.tfrecord'),
    ]:
        num_seq = process_csv_to_tfrecord(csv_path, tfr_path, vocab)
        total_sequences += num_seq
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Total sequences created: {total_sequences:,}")
    print(f"Vocabulary size: {len(vocab.token_to_ids):,}")
    print("\nFiles created:")
    print("  - ./data/vocab.pkl")
    print("  - ./data/train.tfrecord")
    print("  - ./data/val.tfrecord")
    print("  - ./data/test.tfrecord")

if __name__ == "__main__":
    import os
    main()
