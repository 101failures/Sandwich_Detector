"""
Enhanced training script with validation monitoring at each checkpoint
Logs both training and validation metrics for creating train vs val curves
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sandwich_detector import modeling
from sandwich_detector import optimization
import pickle as pkl
import os
import json
import time
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("data_dir", None, "Data directory")
flags.DEFINE_string("bert_config_file", None, "BERT config file")
flags.DEFINE_string("vocab_file", None, "Vocabulary pickle file")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_train_epochs", 10, "Number of training epochs")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("warmup_proportion", 0.1, "Warmup proportion")
flags.DEFINE_integer("log_step_count_steps", 100, "Log every N steps")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "Save checkpoint every N steps")
flags.DEFINE_integer("eval_every_steps", 1000, "Evaluate on validation set every N steps")
flags.DEFINE_integer("random_seed", 42, "Random seed")

# Model parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint")
flags.DEFINE_string("bizdate", "sandwich_detector", "Business date")
flags.DEFINE_string("vocab_dir", "../Model/inter_data/", "Vocab directory")
flags.DEFINE_string("vocab_filename", "vocab", "Vocab filename")
flags.DEFINE_integer("embedding_size", 24, "Address embedding size")
flags.DEFINE_float("address_dropout", 0.5, "Address embedding dropout")


def input_fn(input_file, is_training=True, num_epochs=None):
    """Input function for TFRecord files"""
    name_to_features = {
        "from_address_id": tf.FixedLenFeature([], tf.int64),
        "to_address_id": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64),
        "block_number": tf.FixedLenFeature([], tf.int64),
        "gas": tf.FixedLenFeature([], tf.int64),
        "gas_price": tf.FixedLenFeature([], tf.int64),
        "gas_used": tf.FixedLenFeature([], tf.int64),
        "value": tf.FixedLenFeature([], tf.int64),
        "transaction_index": tf.FixedLenFeature([], tf.int64),
        "sequence_position": tf.FixedLenFeature([], tf.int64),
    }
    
    def _decode_record(record):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example
    
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat(num_epochs)
        d = d.shuffle(buffer_size=10000)
    d = d.map(_decode_record)
    d = d.batch(FLAGS.batch_size)
    
    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()
    return features


def create_model(bert_config, from_addr_id, to_addr_id, transaction_features, label, vocab_size, is_training):
    """Create BERT4ETH model"""
    # Get BERT embeddings
    embedding_table = tf.get_variable(
        name="bert_embeddings",
        shape=[vocab_size, FLAGS.embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    from_embedding = tf.nn.embedding_lookup(embedding_table, from_addr_id)
    to_embedding = tf.nn.embedding_lookup(embedding_table, to_addr_id)
    
    if is_training:
        from_embedding = tf.nn.dropout(from_embedding, keep_prob=1.0 - FLAGS.address_dropout)
        to_embedding = tf.nn.dropout(to_embedding, keep_prob=1.0 - FLAGS.address_dropout)
    
    # Combine embeddings with transaction features
    gas = tf.cast(tf.reshape(transaction_features['gas'], [-1, 1]), tf.float32) / 1e7
    gas_price = tf.cast(tf.reshape(transaction_features['gas_price'], [-1, 1]), tf.float32) / 1e11
    gas_used = tf.cast(tf.reshape(transaction_features['gas_used'], [-1, 1]), tf.float32) / 1e7
    value = tf.cast(tf.reshape(transaction_features['value'], [-1, 1]), tf.float32) / 1e18
    tx_index = tf.cast(tf.reshape(transaction_features['transaction_index'], [-1, 1]), tf.float32) / 100.0
    seq_pos = tf.cast(tf.reshape(transaction_features['sequence_position'], [-1, 1]), tf.float32) / 10.0
    
    combined_input = tf.concat([
        from_embedding, to_embedding,
        gas, gas_price, gas_used, value, tx_index, seq_pos
    ], axis=-1)
    
    # MLP classifier
    hidden1 = tf.layers.dense(combined_input, 256, activation=tf.nn.relu)
    if is_training:
        hidden1 = tf.nn.dropout(hidden1, keep_prob=0.7)
    
    hidden2 = tf.layers.dense(hidden1, 128, activation=tf.nn.relu)
    if is_training:
        hidden2 = tf.nn.dropout(hidden2, keep_prob=0.7)
    
    hidden3 = tf.layers.dense(hidden2, 128, activation=tf.nn.relu)
    if is_training:
        hidden3 = tf.nn.dropout(hidden3, keep_prob=0.7)
    
    logits = tf.layers.dense(hidden3, 4)
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    loss = tf.reduce_mean(loss)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float32))
    
    return loss, predictions, probabilities, accuracy


def evaluate_on_validation(sess, val_loss_op, val_accuracy_op, val_pred_op, val_label_op):
    """Evaluate on full validation set"""
    all_losses = []
    all_accuracies = []
    all_preds = []
    all_labels = []
    
    try:
        while True:
            v_loss, v_acc, v_pred, v_label = sess.run([val_loss_op, val_accuracy_op, val_pred_op, val_label_op])
            all_losses.append(v_loss)
            all_accuracies.append(v_acc)
            all_preds.extend(v_pred.tolist())
            all_labels.extend(v_label.tolist())
    except tf.errors.OutOfRangeError:
        pass
    
    # Calculate overall metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_accuracy = np.mean(all_preds == all_labels)
    avg_loss = np.mean(all_losses)
    
    # Per-class accuracy
    class_accs = {}
    for cls in range(4):
        mask = all_labels == cls
        if np.sum(mask) > 0:
            class_accs[f'class_{cls}_acc'] = float(np.mean(all_preds[mask] == all_labels[mask]))
    
    return {
        'loss': float(avg_loss),
        'accuracy': float(overall_accuracy),
        **class_accs
    }


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    vocab_file = os.path.join(FLAGS.vocab_dir, f"{FLAGS.vocab_filename}.{FLAGS.bizdate}")
    with open(vocab_file, "rb") as f:
        vocab = pkl.load(f)
    
    vocab_size = len(vocab.token_to_ids)
    
    train_file = os.path.join(FLAGS.data_dir, f"sandwich_train.tfrecord.{FLAGS.bizdate}")
    val_file = os.path.join(FLAGS.data_dir, f"sandwich_val.tfrecord.{FLAGS.bizdate}")
    
    num_train_examples = 177386
    num_train_steps = int(num_train_examples / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Random seed: {FLAGS.random_seed}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training steps: {num_train_steps}")
    print(f"Validation evaluation every: {FLAGS.eval_every_steps} steps")
    print(f"{'='*70}\n")
    
    # Build training graph
    tf.reset_default_graph()
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        # Training
        train_features = input_fn(train_file, is_training=True, num_epochs=FLAGS.num_train_epochs)
        train_label = train_features["label"]
        train_from_addr = tf.squeeze(train_features["from_address_id"], axis=-1)
        train_to_addr = tf.squeeze(train_features["to_address_id"], axis=-1)
        train_tx_features = {
            'gas': train_features["gas"],
            'gas_price': train_features["gas_price"],
            'gas_used': train_features["gas_used"],
            'value': train_features["value"],
            'transaction_index': train_features["transaction_index"],
            'sequence_position': train_features["sequence_position"],
        }
        
        train_loss, train_pred, _, train_accuracy = create_model(
            bert_config, train_from_addr, train_to_addr, train_tx_features,
            train_label, vocab_size, is_training=True)
        
        train_op = optimization.create_optimizer(
            train_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
    
    # Build validation graph (separate to avoid training dropout)
    with tf.variable_scope("model", reuse=True):
        val_features = input_fn(val_file, is_training=False, num_epochs=1)
        val_label = val_features["label"]
        val_from_addr = tf.squeeze(val_features["from_address_id"], axis=-1)
        val_to_addr = tf.squeeze(val_features["to_address_id"], axis=-1)
        val_tx_features = {
            'gas': val_features["gas"],
            'gas_price': val_features["gas_price"],
            'gas_used': val_features["gas_used"],
            'value': val_features["value"],
            'transaction_index': val_features["transaction_index"],
            'sequence_position': val_features["sequence_position"],
        }
        
        val_loss, val_pred, _, val_accuracy = create_model(
            bert_config, val_from_addr, val_to_addr, val_tx_features,
            val_label, vocab_size, is_training=False)
    
    saver = tf.train.Saver(max_to_keep=10)
    
    # Training and validation logs
    training_log = []
    validation_log = []
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        print("\nStarting Training with Validation Monitoring...")
        
        step = 0
        start_time = time.time()
        
        try:
            while True:
                # Training step
                _, t_loss, t_acc, t_pred, t_label = sess.run(
                    [train_op, train_loss, train_accuracy, train_pred, train_label])
                step += 1
                
                # Log training metrics
                step_log = {
                    'step': int(step),
                    'epoch': float(step * FLAGS.batch_size / num_train_examples),
                    'train_loss': float(t_loss),
                    'train_accuracy': float(t_acc),
                    'timestamp': time.time()
                }
                training_log.append(step_log)
                
                # Print progress
                if step % FLAGS.log_step_count_steps == 0:
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{num_train_steps}: "
                          f"train_loss={t_loss:.4f}, train_acc={t_acc:.4f}")
                    start_time = time.time()
                
                # Validation evaluation
                if step % FLAGS.eval_every_steps == 0:
                    print(f"\n  Evaluating on validation set...")
                    val_metrics = evaluate_on_validation(sess, val_loss, val_accuracy, val_pred, val_label)
                    
                    val_log = {
                        'step': int(step),
                        'epoch': float(step * FLAGS.batch_size / num_train_examples),
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'timestamp': time.time()
                    }
                    validation_log.append(val_log)
                    
                    print(f"  ✓ Validation: loss={val_metrics['loss']:.4f}, "
                          f"acc={val_metrics['accuracy']:.4f}\n")
                
                # Save checkpoint
                if step % FLAGS.save_checkpoints_steps == 0:
                    checkpoint_path = os.path.join(FLAGS.output_dir, f"model_step_{step}")
                    saver.save(sess, checkpoint_path)
                    
                    # Save logs
                    with open(os.path.join(FLAGS.output_dir, "training_log.json"), 'w') as f:
                        json.dump(training_log, f, indent=2)
                    with open(os.path.join(FLAGS.output_dir, "validation_log.json"), 'w') as f:
                        json.dump(validation_log, f, indent=2)
                    
        except tf.errors.OutOfRangeError:
            print("\nTraining completed!")
        
        # Final validation evaluation
        print("\nFinal validation evaluation...")
        val_metrics = evaluate_on_validation(sess, val_loss, val_accuracy, val_pred, val_label)
        val_log = {
            'step': int(step),
            'epoch': float(FLAGS.num_train_epochs),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'timestamp': time.time()
        }
        validation_log.append(val_log)
        
        # Save final model
        final_checkpoint = os.path.join(FLAGS.output_dir, "model_final")
        saver.save(sess, final_checkpoint)
        
        # Save final logs
        with open(os.path.join(FLAGS.output_dir, "training_log.json"), 'w') as f:
            json.dump(training_log, f, indent=2)
        with open(os.path.join(FLAGS.output_dir, "validation_log.json"), 'w') as f:
            json.dump(validation_log, f, indent=2)
        
        # Also save detailed_training_log.json for backward compatibility
        detailed_log = [{'step': log['step'], 'loss': log['train_loss'], 
                        'accuracy': log['train_accuracy']} for log in training_log]
        with open(os.path.join(FLAGS.output_dir, "detailed_training_log.json"), 'w') as f:
            json.dump(detailed_log, f, indent=2)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total steps: {step}")
        print(f"Final train accuracy: {training_log[-1]['train_accuracy']:.4f}")
        print(f"Final validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    tf.app.run()
