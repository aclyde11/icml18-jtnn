import torch
import torch.nn as nn
from multiprocessing import Pool
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import math, random, sys
from optparse import OptionParser
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm
import pandas as pd
import numpy as np

from joblib import Parallel, delayed

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)


    data = pd.read_csv(opts.train_path, header=None, names=["SMILES"])
    print(data.head())
    print("data length", len(data))
    print("Mapping data to pool")
    data = data.iloc[:5000000,:]

    if rank == 0:
        indicies = np.array(list(range(data.shape[0])))
        indicies = np.array_split(indicies, size)
    else:
        indicies = None

    indicies = comm.scatter(indicies, root=0)
    indicies = print(indicies.shape)
    indicies = list(indicies)

    data = data.iloc[indicies,:]
    print("I am {} with data ".format(rank), data.shape)
    all_data = Parallel(n_jobs=opts.njobs)(delayed(tensorize)(arg[0]) for arg in tqdm(data.itertuples(index=False)))
    "Done"


    num_splits = int(opts.nsplits)
    le = int((len(all_data) + num_splits - 1) / num_splits)
    print("creating splits.")
    for split_id in tqdm(list(range(rank * num_splits, (rank+1) * num_splits))):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
