# Incentive Mechanisms for Fair Subgraph Federated Learning

### About
This codebase evalutes the Pytorch implementation of the paper "[Towards Fair Graph Federated Learning via Incentive Mechanisms](http://arxiv.org/abs/2312.13306)" in the subgraph federated learning setting. Our work aims to evaluate the accuracy and fairness of this framework for node classification under various subgraph partitioning algorithms.

### Usage: How to run the code
```
python main.py 
      --dataset {dataset}
      --partition {graph partitioning algorithm}
      --method {training framework}
      --num_clients {num of clients}
      --seed {random seed}
      --lambda {coefficient of regularization term}
      --outbase {name of the output folder}
      --local_epoch {number of local epochs}
Usage:
--dataset: str, the name of the dataset
--partition: str, the name of the graph partitioning algorithm
--method: str, the name of the training framework
--num_clients: int, the number of clients
--seed: int, random seed of the experiments
--lamb: float, the coefficient of the regularization term
--outbase: str, the file path of the result of the programme, default = './outputs'
--local_epoch, int, number of local epochs
```
##### demo:
```
python mmain.py --dataset Cora --num_clients 10 --partition Metis --method fairfed --seed 1
```

After running the programme, the results are stored in the `./outputs` folder. Or you could modify it in the `--outbase` option.


### Acknowledgement
Some of the implementation is adopted from the following sources [Federated Graph Classification over Non-IID Graphs](https://github.com/Oxfordblue7/GCFL), [Towards Fair Graph Federated Learning via Incentive Mechanisms
](https://github.com/zjunet/FairGraphFL), [Personalized Subgraph Federated Learning] (https://github.com/JinheonBaek/FED-PUB)