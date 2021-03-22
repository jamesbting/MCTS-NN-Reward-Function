import os
import csv
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib import style
from model import Net
#config
config = {
    'save_graphs': False,
    'save_model': False,
    'data_set': '../../data/filtered-dataset-no-header.csv',
    'model': {
        'logs':'logs/model.log',
        'save_location': 'models',
        'input_size': 10,
        'layer_size': 128,
        'dropout_rate': 0.2,
        'learning_rate': 0.0005
    },
    'batch_size': 32,
    'epochs': 5,
    'validation_set_size': 0.1
}

def select_device():
    #set up environment
    if torch.cuda.is_available():
        print("Using CUDA cores")
        return torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 

    else:
        print("Using CPU cores")
        return torch.device("cpu")
        

def load_data_set(filename, validation_set_size, delimiter=','):
    dataset = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            dataset.append(np.array([row]))
    print(f'The size of the entire dataset is {len(dataset)} points')
    val_size = int(len(dataset) * validation_set_size)
    return [dataset[:-val_size], dataset[-val_size:]]
    
def forward_pass(X,y,train = False):
    if train:
        optimizer.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i,0) == torch.argmax(j,0) for i,j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    cross_entropy_loss = loss_function(outputs,y)

    all_linear1_params = torch.cat([x.view(-1) for x in net.fc1.parameters()])
    all_linear2_params = torch.cat([x.view(-1) for x in net.fc2.parameters()])
    l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
    l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)

    loss = cross_entropy_loss + l1_regularization + l2_regularization

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = forward_pass(X.view(-1,config['model']['input_size']).to(device), y.to(device).view(-1,2))
    return val_acc, val_loss

def train(train_X, train_y, net,model_name, model_log_filename):
    with open(model_log_filename,"a+") as f:
        for epoch in range(config['epochs']):
                for i in tqdm(range(0, len(train_X), config['batch_size'])): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:100] ..for now just to dev
                    batch_X = train_X[i:i+config['batch_size']].view(-1,config['model']['input_size'])
                    batch_y = train_y[i:i+config['batch_size']].view(-1,2)
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    acc, loss = forward_pass(batch_X, batch_y, train=True)
                    if i % config['batch_size'] == 0:
                        val_acc, val_loss = test(size=config['batch_size'])
                        f.write(f"{model_name},{round(float(time.time()),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

def create_acc_loss_graph(model_name,save_graphs=False):
    contents = open(model_log_filename,"r").read().split("\n")
    times = []
    accuracies = []
    losses = []
    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")
            times.append(timestamp)
            accuracies.append(acc)
            losses.append(loss)

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    #accuracy graph
    fog = plt.figure()
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

    ax1.plot(times, accuracies, label="test_acc",color="red")
    ax1.axes.get_xaxis().set_ticks([])
    ax1.legend(loc=2)
    ax1.set_title("Training and Valdiation Accuracy")
   
    ax2.plot(times,val_accs, label="val_acc")
    ax2.legend(loc=2)
  

    if save_graphs:
        plt.savefig(f"../logs/graphs/{model_name}-acccuracies.png")
    plt.show()


    #loss graph
    fog = plt.figure()
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
    
    ax1.plot(times,losses, label="test_loss",color="red")
    ax1.axes.get_xaxis().set_ticks([])
    ax1.legend(loc=2)
    ax1.set_title("Training and Validation Loss")
    
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    
    if save_graphs:
         plt.savefig(f"../logs/graphs/{model_name}-POSTMATCHDATA-{INCLUDE_POST_MATCH}-losses.png")
    plt.show()

def create_features_and_labels(dataset):
    features = []
    labels = []
    for point in dataset:
        features.append(np.array([int(x) for x in point[0][0:10]]))
        labels.append(int(point[0][10]))
    
    return [torch.Tensor(features), torch.Tensor(labels)]

def main():
    device = select_device()
    training_set, validation_set = load_data_set(config['data_set'], config['validation_set_size'])
    
    train_X, train_y = create_features_and_labels(training_set)
    test_X, test_Y = create_features_and_labels(validation_set)
    
    layer_size = config['model']['layer_size']
    
    net = Net(config['model']['input_size'],layer_size,config['model']['dropout_rate']).to(device)
    
    optimizer = optim.Adam(net.parameters(),lr = config['model']['learning_rate'])
    model_name = f"model-layersize{layer_size}-{int(time.time())}"
    
    train(train_X, train_y, net,model_name, config['model']['logs'])
    
    print("Hidden Layer size of:",layer_size)
    #make the graphs
    create_acc_loss_graph(model_name)


if __name__ == '__main__':
    main()