# GNN-RT

The GNN-RT can obtain the data-driven representations of molecules through the end-to-end learning with GNN, and predict the retention time with the GNN-learned representations. It takes molecular graph as the input, and the predicted retention time as the output. The GNN architecture is showed as following:

![GNN](https://github.com/Qiong-Yang/GNN-RT/blob/master/support/Figure1.png)

# Motivation

Liquid chromatography (LC) retention time prediction is increasingly getting attention in identification of small molecules, since it supplies information orthogonal to (tandem) MS data for small molecule identification. However the predicted LC retention time of small molecule is not accurate enough for wide adoption in molecular structure.  Hence, we present the GNN-RT method, which is proved to be an effective way to predict small molecule LC  retention time  and improve the accuracy of structural identification of small molecules.

# Depends

Anaconda for python 3.6                      

conda install pytorch

conda install -c rdkit rdkit



# Usage

If you want to train a model based on your in-house database, please put your spectra files in to **data** directory and run **preprocess.py** and  **train.py**.