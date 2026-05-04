#!/bin/bash

datasets=("Cora" "CiteSeer" "PubMed" "Computers" "Photo" "ogbn-arxiv")
partitions=("louvain" "kmeans" "metis" "random")
methods=("central" "selftrain" "fedavg" "fedavg-proto" "fairfed" "fairfed-proto")
num_of_clients=(5 10)
seeds=(42 43 44)

mkdir -p results

for dataset in "${datasets[@]}"; do
    for partition in "${partitions[@]}"; do
        for num_client in "${num_of_clients[@]}"; do
            for method in "${methods[@]}"; do
                for seed in "${seeds[@]}"; do
                    
                    output_file="results/${dataset}_${partition}_${method}_seed${seed}.csv"
                    
                    # Check if file exists so you don't overwrite or rerun completed runs
                    if [ ! -f "$output_file" ]; then
                        echo "Running: $dataset, $partition, $method, seed $seed"
                        
                        python fairfedmotif.py \
                            --dataset "$dataset" \
                            --partition "$partition" \
                            --method "$method" \
                            --seed "$seed" \
                            --split_seed $((seed + 100)) \
                            --output "$output_file"
                    else
                        echo "Skipping: $dataset, $partition, $method, seed $seed (Already exists)"
                    fi

                done
            done
        done
    done
done