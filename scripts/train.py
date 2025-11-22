#!/usr/bin/env python
"""
Wrapper script to run the training module
"""
import sys
import os

# Run the training module as a script
if __name__ == "__main__":
    from sandwich_detector import train
    # The train module uses tf.flags which are automatically parsed
    # No additional argument passing needed
