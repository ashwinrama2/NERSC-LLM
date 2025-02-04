#!/bin/bash

python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# Run Python script to generate common chunks
$python_instance get_prop.py
