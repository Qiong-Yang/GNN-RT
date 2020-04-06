# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:36:10 2019

@author: qy
"""


import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

radius=1
dim=64
hidden_layer=4
extra=65
output_layer=4
batch=32
lr=5e-4
lr_decay=0.9
decay_interval=10
weight_decay=1e-6
iteration=25
update = 'mean'
output = 'mean'


class GraphNeuralNetwork(nn.Module):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim+extra, dim+extra)
                                       for _ in range(output_layer)])
        self.W_property = nn.Linear(dim+extra, 1)

    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [m.shape[0] for m in matrices]
        M = sum(sizes)
        pad_matrices = pad_value + np.zeros((M, M))
        i = 0
        for j, m in enumerate(matrices):
            j = sizes[j]
            pad_matrices[i:i+j, i:i+j] = m
            i += j
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = [torch.sum(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = [torch.mean(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def update(self, xs, A, M, i):
        """Update the node vectors in a graph
        considering their neighboring node vectors (i.e., sum or mean),
        which are non-linear transformed by neural network."""
        hs = torch.relu(self.W_fingerprint[i](xs))
        if update == 'sum':
            return xs + torch.matmul(A, hs)
        if update == 'mean':
            return xs + torch.matmul(A, hs) / (M-1)

    def forward(self, inputs):

        Smiles, fingerprints, adjacencies,descriptors = inputs
        axis = [len(f) for f in fingerprints]
        

        M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        M = torch.unsqueeze(torch.FloatTensor(M), 1)

        fingerprints = torch.cat(fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        
        adjacencies = self.pad(adjacencies, 0)

        """GNN updates the fingerprint vectors."""
        for i in range(hidden_layer):
            fingerprint_vectors = self.update(fingerprint_vectors,
                                              adjacencies, M, i)

        if output == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
        if output == 'mean':
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)
            
        descriptors = torch.stack(descriptors)
        molecular_vectors = torch.cat((molecular_vectors, descriptors),1)
        
        for j in range(output_layer):
            molecular_vectors = torch.relu(self.W_output[j](molecular_vectors))

        molecular_properties = self.W_property(molecular_vectors)

        return Smiles, molecular_properties

    def __call__(self, data_batch, train=True):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties = self.forward(inputs)

        if train:
            loss = F.mse_loss(correct_properties, predicted_properties)
            return loss
        else:
            """Transform the normalized property (i.e., mean 0 and std 1)
            to the unit-based property (e.g., eV and kcal/mol)."""
            correct_properties, predicted_properties = (
                correct_properties.to('cpu').data.numpy(),
                predicted_properties.to('cpu').data.numpy())
            correct_properties, predicted_properties = (
                std * np.concatenate(correct_properties) + mean,
                std * np.concatenate(predicted_properties) + mean)
            return Smiles, correct_properties, predicted_properties


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys, SE_sum = '', [], [], 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            (Smiles, correct_properties, predicted_properties) = self.model(data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_properties)
            Ys.append(predicted_properties)
            SE_sum += sum((correct_properties-predicted_properties)**2)
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        MSE = SE_sum / N
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        return MSE, predictions

    def save_MSEs(self, MSEs, filename):
        with open(filename, 'a') as f:
            f.write(MSEs + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(filename, dtype):
    return [dtype(d).to(device) for d in np.load(filename + '.npy')]


def load_numpy(filename):
    return np.load(filename + '.npy')


def shuffle_dataset(dataset, seed):
     '''
        Task: 
             shuffle a dataset.
        Parameters:
             dataset=load_data(dir_input)
             seed: int (1234)
     '''
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
     '''
        Task: 
             split a dataset.
        Parameters:
             dataset=load_data(dir_input)
             ratio: float (0.2,0.5)
     '''  
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[n:], dataset[:n]
    return dataset_1, dataset_2


if __name__ == "__main__":
    import timeit
    import pandas as pd
    '''
    # if raise a "ValueError"
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    '''
    device = torch.device('cpu') 
    dir_input = 'data/Input/radius' + str(radius) + '/'
    mean = load_numpy(dir_input + 'mean')
    std = load_numpy(dir_input + 'std')
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    n_fingerprint = len(fingerprint_dict)
     
    def load_data(dir_input):
        '''
        Task: 
             Load a dataset.
        Parameters:
             dir_input file stored the molecular Smiles fingerprints,adjacencies 
             properties and descriptors 
        '''
        with open(dir_input + 'Smiles.txt') as f:
            Smiles = f.read().strip().split()
        molecules = load_tensor(dir_input + 'molecules', torch.LongTensor)
        adjacencies = load_numpy(dir_input + 'adjacencies')
        properties = load_tensor(dir_input + 'properties', torch.FloatTensor)
        descriptors = load_tensor(dir_input + 'descriptors', torch.FloatTensor)
        dataset = list(zip(Smiles, molecules, adjacencies,descriptors,properties))
        dataset = shuffle_dataset(dataset, 1234)
        return dataset
    
    dataset = load_data(dir_input)
    dataset_train, dataset_ = split_dataset(dataset, 0.2)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    model = GraphNeuralNetwork().to(device)
    print(model)
    trainer = Trainer(model)
    tester = Tester(model)
    
    """Output files."""
    file_MSEs = 'data/Output/MSEs.txt'
    file_predictions = 'data/Output/predictions.txt'
    file_model = 'data/Output/model.h5'
    #print(file_model)
    MSEs = 'Epoch\tTime(sec)\tLoss_train\tMSE_dev\tMSE_train'
    with open(file_MSEs, 'w') as f:
        f.write(MSEs + '\n')

    """Start training."""
    print('Training...')
    print(MSEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            
        MSE_best = 99999
        
        loss_train = trainer.train(dataset_train)
        MSE_dev = tester.test(dataset_dev)[0]
        MSE_train, predictions_train = tester.test(dataset_train)

        end = timeit.default_timer()
        time = end - start

        MSEs = '\t'.join(map(str, [epoch, time, loss_train,
                                   MSE_dev, MSE_train]))
        tester.save_MSEs(MSEs, file_MSEs)
        tester.save_predictions(predictions_train, file_predictions)
        
        # save best model
        if MSE_dev <= MSE_best:
            MSE_best = MSE_dev
            tester.save_model(model, file_model)

        print(MSEs)
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
    '''
    res = pd.read_table('data/Output/predictions.txt')
    loss = pd.read_table('data/Output/MSEs.txt')
    plt.plot(loss['MSE_test'], color='r')
    plt.plot(loss['MSE_dev'], color='b')
    '''
    
    MSE_test, predictions_test = tester.test(dataset_test)
    tester.save_predictions(predictions_test, 'data/Output/results.txt')
    
    res = pd.read_table('data/Output/results.txt')
    r2 = r2_score(res['Correct'], res['Predict'])
    medae = median_absolute_error(res['Correct'], res['Predict'])
    mae = mean_absolute_error(res['Correct'], res['Predict'])
    median_re = np.median(np.abs(res['Correct'] - res['Predict']) / res['Correct'])
    mean_re=np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct'])
    rmae = np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct']) * 100
    plt.plot(res['Correct'], res['Predict'], '.',alpha=0.3, color = 'blue')
    plt.plot([0,1200], [0,1200], color ='red')
    #plt.plot(label='R2='+str(r2),'MAE='+str(mae),'MRE='+str(mean_re))
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')        
    plt.text(0, 1200, 'MAE='+str(round(mae,4)), fontsize=12)
    plt.text(0, 1100, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(400, 1200, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(400, 1100, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.text(0, 1000, 'R2='+str(round(r2,4)), fontsize=12)
    plt.show()
    plt.show()
    