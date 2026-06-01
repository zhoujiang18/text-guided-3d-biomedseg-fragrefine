#!/bin/bash
set -e

echo "==== Starting prediction ===="
python3 inference.py --input_dir /workspace/inputs/ --output_dir /workspace/outputs/
echo "==== Finished prediction ===="
