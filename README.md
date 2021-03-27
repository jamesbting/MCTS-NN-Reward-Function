# Neural Network Reward Function

A NN based on the following repository: https://github.com/jamesbting/LoL-Predictor

This neural network is to serve as a simulation function for a Monte Carlo Tree Search algorithm. It will save a trained neural network, that the MCTS can use to simulate games. This program assumes the user has a CUDA 11 device that has enough VRAM to train the network. This program can also be trained on a CPU, if no CUDA device is available. 

## Pre-requisites

- Python 3.9.2
- pip 21.0.1

The following Python modules are required as well

- psutil 5.8.0
- numpy 1.20.1
- torch 1.7.1+cu110
- tqdm 4.59.0
- matplotlib 1.20.1

You can install each module by running the command ```pip install <MODULE_NAME>```

This program has not been validated on any other versions of the perquisites. 

## Setting up the program 

After downloading the repository, and installing all the modules, ensure that the file locations in the config dictionary are correct. Configure the hyperparameters of the neural network in the config dictionary, and configure if you want the graphs shown/saved. 

## Running the program

To run the program, ensure you have an editor that can run Jupyter Notebooks. Then, simply run all the cells and the NN will be trained. 

## Results

The trained models are stored by default in the models folder. Each sub-folder is timestamped, and each sub-folder contains 4 graphs, with the training loss, training accuracy, validation loss, validation accuracy and a .pickle file that contains the model binary. By default, the files are named as loss, acc, val_loss, val_acc, model. 

