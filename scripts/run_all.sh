#!/bin/bash

datasets=("Cora" "Photo" "Computers")
partitions=("louvain" "kmeans" "metis" "random")
num_clients=(5 10)
methods=("selftrain" "fedavg" "fedavg-proto")
seeds=(42 43 44)

# ==========================================
# Loop 1: Central Method
# ==========================================
for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running: $dataset, central, seed $seed"
        
        python src/main.py \
            --dataset "$dataset" \
            --method "central" \
            --seed "$seed" \
            --split_seed $((seed + 100)) \
            --local_epoch 200 \
            --model "GIN"
    done
done

# ==========================================
# Loop 2: Federated and Self-Train Methods
# ==========================================
for dataset in "${datasets[@]}"; do
    for num_client in "${num_clients[@]}"; do
        for partition in "${partitions[@]}"; do
            for method in "${methods[@]}"; do
                for seed in "${seeds[@]}"; do
                    echo "Running: $dataset, $partition, $num_client clients, $method, seed $seed"
                    
                    python src/main.py \
                        --dataset "$dataset" \
                        --partition "$partition" \
                        --num_clients "$num_client" \
                        --method "$method" \
                        --seed "$seed" \
                        --split_seed $((seed + 100)) \
                        --local_epoch 200 \
                        -- model "GIN"
                done
            done
        done
    done
done