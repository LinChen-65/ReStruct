import numpy as np
import scipy.sparse as sp
import os
import torch
import sys
import pdb

def main(prefix, threshold, k, seed):

    pos_pairs_offset = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_{threshold}_{seed}.npz"))
    unconnected_pairs_offset = np.load(os.path.join(prefix, "unconnected_pairs_offset.npy"))
    neg_ratings_offset = np.load(os.path.join(prefix, f"neg_ratings_offset_smaller_than_{threshold+1}.npy"))

    train_len = pos_pairs_offset['train'].shape[0]
    val_len = pos_pairs_offset['val'].shape[0]
    test_len = pos_pairs_offset['test'].shape[0]
    pos_len = train_len + val_len + test_len

    if pos_len > neg_ratings_offset.shape[0]:
        indices = np.arange(unconnected_pairs_offset.shape[0])
        assert(indices.shape[0] > pos_len)
        np.random.shuffle(indices)
        makeup = indices[:int((pos_len - neg_ratings_offset.shape[0]) + pos_len*(k-1))]
        neg_ratings_offset = np.concatenate((neg_ratings_offset, unconnected_pairs_offset[makeup]), axis=0)
        assert((pos_len * k) == neg_ratings_offset.shape[0])

    train_len = pos_pairs_offset['train'].shape[0] * k
    val_len = pos_pairs_offset['val'].shape[0] * k
    test_len = pos_pairs_offset['test'].shape[0] * k
    indices = np.arange(neg_ratings_offset.shape[0])
    np.random.shuffle(indices)
    if(k>1):
        np.savez(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_{threshold}_ratio_{k}_{seed}"), 
                    train=neg_ratings_offset[indices[:train_len]],
                    val=neg_ratings_offset[indices[train_len:train_len + val_len]],
                    test=neg_ratings_offset[indices[train_len + val_len:]])
    elif(k==1):
        np.savez(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_{threshold}_{seed}"), 
                    train=neg_ratings_offset[indices[:train_len]],
                    val=neg_ratings_offset[indices[train_len:train_len + val_len]],
                    test=neg_ratings_offset[indices[train_len + val_len:]])

if __name__ == '__main__':
    dataset = sys.argv[1]
    prefix = os.path.join("./data_recommendation/", dataset)
    
    seed = int(sys.argv[2])
    np.random.seed(seed)
    
    k = int(sys.argv[3])
    threshold = 2

    main(prefix, threshold, k, seed)
