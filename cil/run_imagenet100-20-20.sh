#!/bin/bash

# For ImageNet-100 dataset; 20 classes/task, 5 tasks
python main.py \
    --config-path configs/class \
    --config-name imagenet100_20-20.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/imagenet100.yaml"