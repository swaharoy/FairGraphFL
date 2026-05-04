#!/bin/bash

DATASETS=("Cora" "CiteSeer" "PubMed" "Computers" "Photo" "ogbn-arxiv")
PARTITIONS=("louvain" "kmeans" "metis" "random")


NUM_CLIENTS=10

SEED=1

for dataset in "${DATASETS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    
    echo "======================================"
    echo "Running dataset=$dataset partition=$partition"
    echo "======================================"

    python src/main.py \
      --dataset "$dataset" \
      --num_clients "$NUM_CLIENTS" \
      --partition "$partition" \
      --seed "$SEED"

  done
done

echo "All runs completed."