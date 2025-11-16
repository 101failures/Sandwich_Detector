"""
CRF (Conditional Random Field) layer for sequence labeling.
Enforces valid sandwich attack transition constraints.
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_addons as tfa

class CRF:
    """
    Conditional Random Field layer for sequence labeling.
    Learns transition probabilities between labels.
    """
    
    def __init__(self, num_labels):
        self.num_labels = num_labels
        
    def __call__(self, inputs, sequence_lengths, labels=None, training=True):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_labels) - logits from classification layer
            sequence_lengths: (batch_size,) - actual length of each sequence
            labels: (batch_size, max_seq_len) - true labels (only for training)
            training: bool - whether in training mode
            
        Returns:
            If training:
                loss: scalar tensor
                predictions: (batch_size, max_seq_len)
            Else:
                predictions: (batch_size, max_seq_len)
        """
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            # Transition matrix: (num_labels, num_labels)
            # transitions[i, j] = score of transitioning from label i to label j
            self.transitions = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.zeros_initializer()
            )
            
            # Apply sandwich attack constraints
            # Create mask for invalid transitions (set to very negative value)
            constraint_mask = self._create_constraint_mask()
            constrained_transitions = self.transitions * constraint_mask
            
            if training and labels is not None:
                # Compute CRF loss using tensorflow-addons
                log_likelihood, _ = tfa.text.crf_log_likelihood(
                    inputs=inputs,
                    tag_indices=labels,
                    sequence_lengths=sequence_lengths,
                    transition_params=constrained_transitions
                )
                loss = tf.reduce_mean(-log_likelihood)
                
                # Viterbi decoding for predictions
                viterbi_sequences, _ = tfa.text.crf_decode(
                    potentials=inputs,
                    transition_params=constrained_transitions,
                    sequence_length=sequence_lengths
                )
                
                return loss, viterbi_sequences
            else:
                # Inference: Viterbi decoding only
                viterbi_sequences, _ = tfa.text.crf_decode(
                    potentials=inputs,
                    transition_params=constrained_transitions,
                    sequence_length=sequence_lengths
                )
                
                return viterbi_sequences
    
    def _create_constraint_mask(self):
        """
        Create STRICT mask for sandwich attack transition constraints.
        
        Labels: 0=non-sandwich, 1=frontrun, 2=victim, 3=backrun
        
        STRICT RULES (sandwich must be complete):
        - non-sandwich can ONLY go to: non-sandwich or frontrun (start of attack)
        - frontrun MUST go to: victim or backrun (no isolated frontrun)
        - victim MUST go to: backrun (victim must be sandwiched)
        - backrun can ONLY go to: non-sandwich (attack complete, reset)
        
        This enforces that victims CANNOT exist without frontrun before and backrun after.
        """
        # Start with all transitions INVALID (-10.0 = strong penalty)
        mask = np.ones([self.num_labels, self.num_labels], dtype=np.float32) * -10.0
        
        # Define ONLY valid transitions (set to 0.0 = no penalty)
        valid_transitions = [
            (0, 0),  # non-sandwich → non-sandwich (normal activity continues)
            (0, 1),  # non-sandwich → frontrun (attack starts)
            (1, 2),  # frontrun → victim (REQUIRED - no sandwich without victim)
            (2, 3),  # victim → backrun (sandwich completes)
            (3, 0),  # backrun → non-sandwich (attack ends)
        ]
        
        for from_label, to_label in valid_transitions:
            mask[from_label, to_label] = 0.0
        
        return tf.constant(mask, dtype=tf.float32)
