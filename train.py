
import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import preprocess as pp
import pickle

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return Smiles,molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles,molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles,predicted_values, correct_values

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model
    def test_regressor(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values)
            Ys.append(predicted_values)
            
            SAE += sum(np.abs(predicted_values-correct_values))
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        #MSE = SE_sum / N
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        MAEs = SAE / N  # mean absolute error.
        return MAEs,predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)
if __name__ == "__main__":
    radius=1
    dim=48
    layer_hidden=6
    layer_output=6
    batch_train=32
    batch_test=32
    lr=1e-4
    lr_decay=0.85
    decay_interval=10
    iteration=200
    N=5000
    path='/data/'
    dataname='SMRT'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    dataset_train = pp.create_dataset('SMRT_train_set.txt',path,dataname)
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = pp.create_dataset('SMRT_test_set.txt',path,dataname)    
    lr, lr_decay = map(float, [lr, lr_decay])
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)
    #print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')    
    print('-'*100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)
    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            N, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)
    file_MAEs = path+'MAEs'+'.txt'
    file_test_result  =  path+ 'test_prediction'+ '.txt'
    file_predictions =  path+'train_prediction' +'.txt'
    file_model =   path+'model'+'.h5'
    result  = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_dev\tMAE_test'
    with open(file_MAEs , 'w') as f:
        f.write(result  + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')
    np.random.seed(1234)       
    start = timeit.default_timer()

    for epoch in range(iteration ):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        model.train()
        loss_train = trainer.train(dataset_train)
        MAE_best=9999999
        model.eval()
        MAE_train,predictions_train  = tester.test_regressor(dataset_train)
        MAE_dev = tester.test_regressor(dataset_dev)[0]
        MAE_test = tester.test_regressor(dataset_test)[0]

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration  / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result )

        results  = '\t'.join(map(str, [epoch, time, loss_train,MAE_train,
                                     MAE_dev, MAE_test]))
        tester.save_MAEs(results , file_MAEs )
        if MAE_dev <= MAE_best:
            MAE_best = MAE_dev
            tester.save_model(model, file_model)
        print(results )
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    loss = pd.read_table(file_MAEs)
    plt.plot(loss['MAE_train'], color='r',label='MSE of train set')
    plt.plot(loss['MAE_dev'], color='b',label='MSE of validation set')
    plt.plot(loss['MAE_test'], color='y',label='MSE of test set')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(path+'loss.tif',dpi=300)
    plt.show()
    predictions_train  = tester.test_regressor(dataset_train)[1]
    tester.save_predictions(predictions_train, file_predictions )
    predictions_test  = tester.test_regressor(dataset_test)[1]
    tester.save_predictions(predictions_test , file_test_result )
    res  = pd.read_table(file_test_result)
    r2 = r2_score(res ['Correct'], res ['Predict'])
    mae = mean_absolute_error(res ['Correct'], res ['Predict'])
    medae = median_absolute_error(res ['Correct'], res ['Predict'])
    rmae = np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct']) * 100
    median_re = np.median(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    mean_re=np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    plt.plot(res ['Correct'], res ['Predict'], '.', color = 'blue')
    plt.plot([0,2000], [0,2000], color ='red')
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')        
    plt.text(0,2000, 'R2='+str(round(r2,4)), fontsize=12)
    plt.text(800,2000,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(0, 1800, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(800, 1800, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(0, 1600, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.savefig( path+'c-p.tif',dpi=300)
    plt.show()
  
