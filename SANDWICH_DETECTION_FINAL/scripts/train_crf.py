"""
Train BERT4ETH model for sandwich attack detection.
Fine-tunes pretrained model with token-level classification (4 classes).
"""

import os
import sys
import pickle as pkl
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Configure GPU memory to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# Add Model and parent directory for imports
sys.path.append("../../Model")
sys.path.append("..")
import modeling
import optimization
from crf_layer import CRF
from vocab import SimpleVocab

flags = tf.flags
FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("bert_config_file", "../../Model/bert_config.json", "BERT config file")
flags.DEFINE_string("init_checkpoint", "../../latest_checkpoint/model_752000", "Pretrained checkpoint - will load what exists, init new features randomly")
flags.DEFINE_string("vocab_file", "../data/vocab.pkl", "Vocabulary file")
flags.DEFINE_string("train_file", "../data/train.tfrecord", "Training TFRecord")
flags.DEFINE_string("val_file", "../data/val.tfrecord", "Validation TFRecord")
flags.DEFINE_string("output_dir", "./FINALMODEL", "Output directory for checkpoints")

flags.DEFINE_integer("max_seq_length", 100, "Maximum sequence length")
flags.DEFINE_integer("batch_size", 8, "Batch size - restored to 8 now that checkpoint not loaded")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_integer("num_train_steps", 10000, "Number of training steps")
flags.DEFINE_integer("warmup_steps", 2000, "Number of warmup steps")
flags.DEFINE_integer("save_every", 500, "Save checkpoint every N steps - MORE FREQUENT")
flags.DEFINE_integer("eval_every", 500, "Evaluate on validation every N steps")

NUM_CLASSES = 4  # non-sandwich, frontrun, victim, backrun

def _decode_record(record, name_to_features):
    """Decode TFRecord to features."""
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example

def input_fn(input_file, is_training, batch_size):
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
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=1000)
    
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
                 input_logs_counts, input_counts, input_mask, label_ids, is_training, reuse=False):
    """Create the model for training."""
    
    # BERT encoder
    with tf.variable_scope("bert", reuse=reuse):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
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
    
    # Classification head - output logits for CRF
    with tf.variable_scope("sandwich_classifier", reuse=reuse):
        output_layer = tf.layers.dense(
            sequence_output,
            NUM_CLASSES,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="output"
        )
        
        logits = output_layer
    
    # CRF layer for sequence labeling
    # Enforces transition constraints: frontrun → victim → backrun
    sequence_lengths = tf.reduce_sum(input_mask, axis=1)
    crf = CRF(num_labels=NUM_CLASSES)
    
    if is_training:
        # Training: compute CRF loss and get predictions
        loss, predictions = crf(logits, sequence_lengths, label_ids, training=True)
    else:
        # Inference: just get predictions via Viterbi
        predictions = crf(logits, sequence_lengths, training=False)
        loss = tf.constant(0.0)
    
    # For compatibility, create dummy probabilities
    probabilities = tf.nn.softmax(logits, axis=-1)
    
    return loss, predictions, probabilities, logits

def compute_metrics(predictions, labels, mask):
    """Compute accuracy and per-class metrics."""
    # Flatten and apply mask
    preds_flat = predictions[mask == 1]
    labels_flat = labels[mask == 1]
    
    accuracy = np.mean(preds_flat == labels_flat)
    
    # Per-class metrics
    metrics = {'accuracy': accuracy}
    for class_id in range(NUM_CLASSES):
        class_mask = labels_flat == class_id
        if class_mask.sum() > 0:
            class_acc = np.mean(preds_flat[class_mask] == labels_flat[class_mask])
            metrics[f'class_{class_id}_acc'] = class_acc
            metrics[f'class_{class_id}_count'] = class_mask.sum()
    
    return metrics

def evaluate(sess, val_iterator, val_features, predictions_op, labels_op, mask_op):
    """Evaluate on validation set."""
    sess.run(val_iterator.initializer)
    
    all_preds = []
    all_labels = []
    all_masks = []
    
    try:
        while True:
            preds, labels, mask = sess.run([predictions_op, labels_op, mask_op])
            all_preds.append(preds)
            all_labels.append(labels)
            all_masks.append(mask)
    except tf.errors.OutOfRangeError:
        pass
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_masks)
    
    return metrics

def main(_):
    print("="*80)
    print("BERT4ETH Sandwich Detection - Training")
    print("="*80)
    
    # Load vocab
    print(f"\nLoading vocabulary from {FLAGS.vocab_file}")
    with open(FLAGS.vocab_file, 'rb') as f:
        vocab = pkl.load(f)
    print(f"Vocabulary size: {len(vocab.token_to_ids):,}")
    
    # Load BERT config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    print(f"\nBERT config:")
    print(f"  Hidden size: {bert_config.hidden_size}")
    print(f"  Num layers: {bert_config.num_hidden_layers}")
    print(f"  Num attention heads: {bert_config.num_attention_heads}")
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Build training graph
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)
    
    train_dataset = input_fn(FLAGS.train_file, is_training=True, batch_size=FLAGS.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_features = train_iterator.get_next()
    
    loss, predictions, probabilities, logits = create_model(
        bert_config,
        train_features["input_ids"],
        train_features["input_positions"],
        train_features["input_io_flags"],
        train_features["input_values"],
        train_features["input_gas_prices"],
        train_features["input_tx_indices"],
        train_features["input_gas_useds"],
        train_features["input_logs_counts"],
        train_features["input_counts"],
        train_features["input_mask"],
        train_features["label_ids"],
        is_training=True
    )
    
    # Optimizer
    train_op = optimization.create_optimizer(
        loss, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.warmup_steps, use_tpu=False
    )
    
    # Load pretrained weights
    tvars = tf.trainable_variables()
    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, FLAGS.init_checkpoint
        )
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
        
        print("\nLoaded pretrained weights from:", FLAGS.init_checkpoint)
        print(f"Initialized {len(initialized_variable_names)} variables")
    
    # Build validation graph (reuse variables)
    val_dataset = input_fn(FLAGS.val_file, is_training=False, batch_size=FLAGS.batch_size)
    val_iterator = val_dataset.make_initializable_iterator()
    val_features = val_iterator.get_next()
    
    _, val_predictions, _, _ = create_model(
        bert_config,
        val_features["input_ids"],
        val_features["input_positions"],
        val_features["input_io_flags"],
        val_features["input_values"],
        val_features["input_gas_prices"],
        val_features["input_tx_indices"],
        val_features["input_gas_useds"],
        val_features["input_logs_counts"],
        val_features["input_counts"],
        val_features["input_mask"],
        val_features["label_ids"],
        is_training=False,
        reuse=True
    )
    
    # Saver
    saver = tf.train.Saver(max_to_keep=5)
    
    # Training configuration
    print("\n" + "="*80)
    print("Training Configuration:")
    print("="*80)
    print(f"  Batch size: {FLAGS.batch_size}")
    print(f"  Learning rate: {FLAGS.learning_rate}")
    print(f"  Training steps: {FLAGS.num_train_steps:,}")
    print(f"  Warmup steps: {FLAGS.warmup_steps:,}")
    print(f"  Save every: {FLAGS.save_every} steps")
    print(f"  Eval every: {FLAGS.eval_every} steps")
    print(f"  Using CRF layer with transition constraints (frontrun → victim → backrun)")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    # Configure GPU memory growth to prevent OOM
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # Use 80% max GPU memory
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        
        # Try to restore from checkpoint if exists
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        if checkpoint_path:
            print(f"\nRestoring from checkpoint: {checkpoint_path}")
            saver.restore(sess, checkpoint_path)
            # Extract step number from checkpoint path (e.g., model_step_1000 -> 1000)
            try:
                if 'step_' in checkpoint_path:
                    start_step = int(checkpoint_path.split('step_')[-1])
                else:
                    start_step = 0  # best_model has no step number
            except:
                start_step = 0
            print(f"Resuming from step {start_step}")
        else:
            start_step = 0
            print("\nNo checkpoint found, starting from scratch")
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 5  # Early stopping: stop if no improvement for 5 evaluations
        
        for step in range(start_step, FLAGS.num_train_steps):
            _, loss_val = sess.run([train_op, loss])
            
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{FLAGS.num_train_steps} - Loss: {loss_val:.4f}")
            
            # Evaluate on validation
            if (step + 1) % FLAGS.eval_every == 0:
                print(f"\n--- Validation at step {step+1} ---")
                val_metrics = evaluate(
                    sess, val_iterator, val_features, 
                    val_predictions, val_features["label_ids"], val_features["input_mask"]
                )
                
                print(f"  Overall accuracy: {val_metrics['accuracy']:.4f}")
                for class_id in range(NUM_CLASSES):
                    class_name = ['non-sandwich', 'frontrun', 'victim', 'backrun'][class_id]
                    if f'class_{class_id}_acc' in val_metrics:
                        print(f"  {class_name}: {val_metrics[f'class_{class_id}_acc']:.4f} "
                              f"(n={val_metrics[f'class_{class_id}_count']})")
                
                # Save if best
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0  # Reset patience
                    save_path = os.path.join(FLAGS.output_dir, "best_model")
                    saver.save(sess, save_path)
                    print(f"  ✓ New best model saved (acc={best_val_acc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement (patience: {patience_counter}/{patience})")
                    if patience_counter >= patience:
                        print(f"\n⚠ Early stopping triggered after {step+1} steps")
                        print(f"Best validation accuracy: {best_val_acc:.4f}")
                        final_path = os.path.join(FLAGS.output_dir, "model_final_early_stop")
                        saver.save(sess, final_path)
                        break
                print()
            
            # Save checkpoint
            if (step + 1) % FLAGS.save_every == 0:
                save_path = os.path.join(FLAGS.output_dir, f"model_step_{step+1}")
                saver.save(sess, save_path)
                print(f"✓ Checkpoint saved: {save_path}")
        
        # Final save
        final_path = os.path.join(FLAGS.output_dir, "model_final")
        saver.save(sess, final_path)
        print(f"\n✓ Final model saved: {final_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved in: {FLAGS.output_dir}")

if __name__ == "__main__":
    tf.app.run()
