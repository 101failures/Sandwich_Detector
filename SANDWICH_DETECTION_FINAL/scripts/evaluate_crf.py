"""
Evaluate CRF model on test set with comprehensive metrics.
"""

import os
import sys
import pickle as pkl
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import tensorflow_addons as tfa

# Add Model and parent directory for imports
sys.path.append("../../Model")
sys.path.append("..")
import modeling
from crf_layer import CRF
from vocab import SimpleVocab

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", "../../Model/bert_config.json", "BERT config file")
flags.DEFINE_string("checkpoint", "./FINALMODEL/best_model", "Checkpoint to evaluate")
flags.DEFINE_string("vocab_file", "../data/vocab.pkl", "Vocabulary file")
flags.DEFINE_string("test_file", "../data/test.tfrecord", "Test TFRecord")
flags.DEFINE_integer("max_seq_length", 100, "Maximum sequence length")
flags.DEFINE_integer("batch_size", 8, "Batch size")

NUM_CLASSES = 4
CLASS_NAMES = ['non-sandwich', 'frontrun', 'victim', 'backrun']

def _decode_record(record, name_to_features):
    """Decode TFRecord."""
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example

def input_fn(input_file, batch_size):
    """Create input pipeline."""
    name_to_features = {
        "address": tf.FixedLenFeature([1], tf.int64),
        "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_positions": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_counts": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_io_flags": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_values": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_gas_prices": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_tx_indices": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_gas_useds": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_priority_fees": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_logs_counts": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_nonces": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_tx_types": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
    }
    
    d = tf.data.TFRecordDataset(input_file)
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=4,
            drop_remainder=False
        )
    )
    return d

def create_model(bert_config, input_ids, input_positions, input_io_flags,
                 input_values, input_gas_prices, input_tx_indices, input_gas_useds,
                 input_logs_counts, input_counts, input_mask, label_ids):
    """Create model for evaluation."""
    
    # BERT encoder
    with tf.variable_scope("bert"):
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_positions=input_positions,
            input_io_flags=input_io_flags,
            input_amounts=input_values,
            input_gas_prices=input_gas_prices,
            input_tx_indices=input_tx_indices,
            input_gas_useds=input_gas_useds,
            input_logs_counts=input_logs_counts,
            input_counts=input_counts,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False
        )
        
        sequence_output = model.get_sequence_output()
    
    # Classification head
    with tf.variable_scope("sandwich_classifier"):
        output_layer = tf.layers.dense(
            sequence_output,
            NUM_CLASSES,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="output"
        )
        logits = output_layer
    
    # CRF layer
    sequence_lengths = tf.reduce_sum(input_mask, axis=1)
    crf = CRF(num_labels=NUM_CLASSES)
    predictions = crf(logits, sequence_lengths, training=False)
    
    return predictions

def compute_sequence_metrics(predictions, labels, masks):
    """
    Compute sequence-level metrics: how many complete sandwiches were correctly identified.
    """
    num_sequences = len(predictions)
    correct_sandwiches = 0
    total_sandwiches = 0
    
    for pred, label, mask in zip(predictions, labels, masks):
        # Get valid tokens
        valid_len = np.sum(mask)
        pred_valid = pred[:valid_len]
        label_valid = label[:valid_len]
        
        # Check if this sequence contains a sandwich in ground truth
        has_sandwich = (1 in label_valid) or (2 in label_valid) or (3 in label_valid)
        
        if has_sandwich:
            total_sandwiches += 1
            
            # Check if prediction correctly identifies the complete sandwich pattern
            # (at least one frontrun, victim, and backrun in correct order)
            pred_has_frontrun = 1 in pred_valid
            pred_has_victim = 2 in pred_valid
            pred_has_backrun = 3 in pred_valid
            
            if pred_has_frontrun and pred_has_victim and pred_has_backrun:
                # Check order: frontrun before victim before backrun
                frontrun_idx = np.where(pred_valid == 1)[0]
                victim_idx = np.where(pred_valid == 2)[0]
                backrun_idx = np.where(pred_valid == 3)[0]
                
                if len(frontrun_idx) > 0 and len(victim_idx) > 0 and len(backrun_idx) > 0:
                    if frontrun_idx[0] < victim_idx[0] < backrun_idx[-1]:
                        correct_sandwiches += 1
    
    sequence_recall = correct_sandwiches / total_sandwiches if total_sandwiches > 0 else 0.0
    return sequence_recall, correct_sandwiches, total_sandwiches

def main(_):
    print("="*80)
    print("CRF Model Evaluation")
    print("="*80)
    print()
    
    # Load vocab
    print(f"Loading vocabulary from {FLAGS.vocab_file}")
    with open(FLAGS.vocab_file, 'rb') as f:
        vocab = pkl.load(f)
    print("Vocabulary loaded")
    print()
    
    # Load BERT config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    print("BERT config loaded")
    print()
    
    # Build model
    print("Building model...")
    dataset = input_fn(FLAGS.test_file, FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()
    features = iterator.get_next()
    
    predictions = create_model(
        bert_config,
        features["input_ids"],
        features["input_positions"],
        features["input_io_flags"],
        features["input_values"],
        features["input_gas_prices"],
        features["input_tx_indices"],
        features["input_gas_useds"],
        features["input_logs_counts"],
        features["input_counts"],
        features["input_mask"],
        features["label_ids"]
    )
    
    # Restore checkpoint
    saver = tf.train.Saver()
    
    print(f"Loading checkpoint: {FLAGS.checkpoint}")
    print()
    
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint)
        sess.run(iterator.initializer)
        
        all_predictions = []
        all_labels = []
        all_masks = []
        
        print("Running evaluation...")
        batch_count = 0
        try:
            while True:
                preds, labels, masks = sess.run([
                    predictions,
                    features["label_ids"],
                    features["input_mask"]
                ])
                all_predictions.append(preds)
                all_labels.append(labels)
                all_masks.append(masks)
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processed {batch_count} batches...")
        except tf.errors.OutOfRangeError:
            pass
        
        print(f"Processed {batch_count} batches total")
        print()
        
        # Concatenate results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        # Flatten for token-level metrics
        preds_flat = all_predictions[all_masks == 1]
        labels_flat = all_labels[all_masks == 1]
        
        print("="*80)
        print("TOKEN-LEVEL METRICS")
        print("="*80)
        print()
        
        # Overall accuracy
        accuracy = np.mean(preds_flat == labels_flat)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print()
        
        # Per-class metrics
        print("Per-Class Performance:")
        print("-" * 60)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_flat, preds_flat, labels=[0, 1, 2, 3], average=None
        )
        
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:15s} - P: {precision[i]:.4f}  R: {recall[i]:.4f}  "
                  f"F1: {f1[i]:.4f}  (n={support[i]})")
        
        print()
        print("Macro Averages:")
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, labels=[0, 1, 2, 3], average='macro'
        )
        print(f"  Precision: {macro_p:.4f}")
        print(f"  Recall:    {macro_r:.4f}")
        print(f"  F1-Score:  {macro_f1:.4f}")
        print()
        
        # Confusion matrix
        print("Confusion Matrix:")
        print("-" * 60)
        cm = confusion_matrix(labels_flat, preds_flat, labels=[0, 1, 2, 3])
        
        # Print header
        print(f"{'':15s}", end="")
        for name in CLASS_NAMES:
            print(f"{name[:10]:>12s}", end="")
        print()
        print("-" * 60)
        
        # Print rows
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:15s}", end="")
            for j in range(NUM_CLASSES):
                print(f"{cm[i, j]:>12d}", end="")
            print()
        print()
        
        # Save confusion matrix as PNG
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max() / 2 else "black",
                              fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title('Confusion Matrix - CRF Model on Test Set', fontsize=15, fontweight='bold', pad=20)
        
        fig.tight_layout()
        confusion_matrix_path = './confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to: {confusion_matrix_path}")
        plt.close()
        print()
        
        # Sequence-level metrics
        print("="*80)
        print("SEQUENCE-LEVEL METRICS")
        print("="*80)
        print()
        
        seq_recall, correct, total = compute_sequence_metrics(
            all_predictions, all_labels, all_masks
        )
        
        print(f"Complete Sandwich Recovery Rate: {seq_recall:.4f}")
        print(f"  (Correctly identified {correct}/{total} complete sandwiches)")
        print()
        
        # Summary
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print()
        print(f"✓ Token-level accuracy: {accuracy:.4f}")
        print(f"✓ Frontrun F1:  {f1[1]:.4f}")
        print(f"✓ Victim F1:    {f1[2]:.4f}")
        print(f"✓ Backrun F1:   {f1[3]:.4f}")
        print(f"✓ Macro F1:     {macro_f1:.4f}")
        print(f"✓ Sandwich recovery: {seq_recall:.4f}")
        print()
        print("="*80)

if __name__ == "__main__":
    tf.app.run()
