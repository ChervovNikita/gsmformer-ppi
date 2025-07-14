import os
import torch
import glob
import numpy as np
import random
import math
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset as Dataset_n
from torch.utils.data import DataLoader  # <-- from standard PyTorch
from torch_geometric.data import Data, Batch
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--npy_file', type=str, default='../masif_features/new_train.npy')
# parser.add_argument('--cv_fold', type=int, default=5)
# parser.add_argument('--cv_fold_idx', type=int, default=0)
# args = parser.parse_args
# args, unknown = parser.parse_known_args()    # remove after test (David) 

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def bump(g):
    return g

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir, desc=True):
        self.npy_ar = np.load(npy_file)
        self.processed_dir = processed_dir
        self.protein_1 = self.npy_ar[:,2]
        self.protein_2 = self.npy_ar[:,5]
        self.label = self.npy_ar[:,6].astype(float)
        self.n_samples = self.npy_ar.shape[0]
        self.desc = desc

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
        prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
        
        # print(prot_1)
        prot_1 = torch.load(glob.glob(prot_1)[0], weights_only=False)
        # print(prot_2)
        prot_2 = torch.load(glob.glob(prot_2)[0], weights_only=False)
        prot_1 = bump(prot_1)
        prot_2 = bump(prot_2)
        prot_1.x = prot_1.x.to(torch.float32)
        prot_2.x = prot_2.x.to(torch.float32)

        if not self.desc:
            return prot_1, prot_2, torch.tensor(self.label[index], dtype=torch.float32), torch.zeros((1, 80)), torch.zeros((1, 80)), torch.zeros((1, 80)), torch.zeros((1, 80))

        # print(self.protein_1[index])
        # assert False

        # all_prots = os.listdir(os.path.join(self.processed_dir, 'masif_descriptors'))
        # real_name = None
        # for name in all_prots:
        #     if name.startswith(self.protein_1[index].split('_')[0]):
        #         real_name = name
        #         break
        prot_1_masif_straight = os.path.join(self.processed_dir, 'masif_descriptors', self.protein_1[index], 'desc_straight.npy')
        prot_1_masif_flipped = os.path.join(self.processed_dir, 'masif_descriptors', self.protein_1[index], 'desc_flipped.npy')
        # real_name = None
        # for name in all_prots:
        #     if name.startswith(self.protein_2[index].split('_')[0]):
        #         real_name = name
        #         break
        prot_2_masif_straight = os.path.join(self.processed_dir, 'masif_descriptors', self.protein_2[index], 'desc_straight.npy')
        prot_2_masif_flipped = os.path.join(self.processed_dir, 'masif_descriptors', self.protein_2[index], 'desc_flipped.npy')

        prot_1_masif_straight = torch.tensor(np.load(prot_1_masif_straight))
        prot_1_masif_flipped = torch.tensor(np.load(prot_1_masif_flipped))
        prot_2_masif_straight = torch.tensor(np.load(prot_2_masif_straight))
        prot_2_masif_flipped = torch.tensor(np.load(prot_2_masif_flipped))
        return (
            prot_1,
            prot_2,
            torch.tensor(self.label[index], dtype=torch.float32),
            prot_1_masif_straight,
            prot_1_masif_flipped,
            prot_2_masif_straight,
            prot_2_masif_flipped
        )

def collate_fn(batch):
    prot_1s = [item[0] for item in batch]
    prot_2s = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    p1_straight = [item[3] for item in batch]
    p1_flipped = [item[4] for item in batch]
    p2_straight = [item[5] for item in batch]
    p2_flipped = [item[6] for item in batch]

    max_len_p1_straight = max(x.size(0) for x in p1_straight)
    max_len_p1_flipped = max(x.size(0) for x in p1_flipped)
    max_len_p2_straight = max(x.size(0) for x in p2_straight)
    max_len_p2_flipped = max(x.size(0) for x in p2_flipped)

    p1_straight_padded = []
    for x in p1_straight:
        tmp = torch.zeros((max_len_p1_straight, x.size(1)))
        tmp[:x.size(0)] = x
        p1_straight_padded.append(tmp)

    p1_flipped_padded = []
    for x in p1_flipped:
        tmp = torch.zeros((max_len_p1_flipped, x.size(1)))
        tmp[:x.size(0)] = x
        p1_flipped_padded.append(tmp)

    p2_straight_padded = []
    for x in p2_straight:
        tmp = torch.zeros((max_len_p2_straight, x.size(1)))
        tmp[:x.size(0)] = x
        p2_straight_padded.append(tmp)

    p2_flipped_padded = []
    for x in p2_flipped:
        tmp = torch.zeros((max_len_p2_flipped, x.size(1)))
        tmp[:x.size(0)] = x
        p2_flipped_padded.append(tmp)

    p1_straight_padded = torch.stack(p1_straight_padded, dim=0)
    p1_flipped_padded = torch.stack(p1_flipped_padded, dim=0)
    p2_straight_padded = torch.stack(p2_straight_padded, dim=0)
    p2_flipped_padded = torch.stack(p2_flipped_padded, dim=0)

    batch_1 = Batch.from_data_list(prot_1s)
    batch_2 = Batch.from_data_list(prot_2s)
    return batch_1, batch_2, labels, p1_straight_padded, p1_flipped_padded, p2_straight_padded, p2_flipped_padded

def collate_fn_unpadded(batch):
    """
    New collate function that returns unpadded descriptors with length information.
    For use with GCNN_geom_transformer to avoid double padding.
    """
    prot_1s = [item[0] for item in batch]
    prot_2s = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    p1_straight = [item[3] for item in batch]
    p1_flipped = [item[4] for item in batch]
    p2_straight = [item[5] for item in batch]
    p2_flipped = [item[6] for item in batch]

    # Collect actual sequence lengths
    p1_straight_lengths = torch.tensor([x.size(0) for x in p1_straight], dtype=torch.long)
    p1_flipped_lengths = torch.tensor([x.size(0) for x in p1_flipped], dtype=torch.long)
    p2_straight_lengths = torch.tensor([x.size(0) for x in p2_straight], dtype=torch.long)
    p2_flipped_lengths = torch.tensor([x.size(0) for x in p2_flipped], dtype=torch.long)

    # Find maximum lengths
    max_len_p1_straight = max(x.size(0) for x in p1_straight)
    max_len_p1_flipped = max(x.size(0) for x in p1_flipped)
    max_len_p2_straight = max(x.size(0) for x in p2_straight)
    max_len_p2_flipped = max(x.size(0) for x in p2_flipped)

    # Pad sequences
    p1_straight_padded = []
    for x in p1_straight:
        tmp = torch.zeros((max_len_p1_straight, x.size(1)))
        tmp[:x.size(0)] = x
        p1_straight_padded.append(tmp)

    p1_flipped_padded = []
    for x in p1_flipped:
        tmp = torch.zeros((max_len_p1_flipped, x.size(1)))
        tmp[:x.size(0)] = x
        p1_flipped_padded.append(tmp)

    p2_straight_padded = []
    for x in p2_straight:
        tmp = torch.zeros((max_len_p2_straight, x.size(1)))
        tmp[:x.size(0)] = x
        p2_straight_padded.append(tmp)

    p2_flipped_padded = []
    for x in p2_flipped:
        tmp = torch.zeros((max_len_p2_flipped, x.size(1)))
        tmp[:x.size(0)] = x
        p2_flipped_padded.append(tmp)

    p1_straight_padded = torch.stack(p1_straight_padded, dim=0)
    p1_flipped_padded = torch.stack(p1_flipped_padded, dim=0)
    p2_straight_padded = torch.stack(p2_straight_padded, dim=0)
    p2_flipped_padded = torch.stack(p2_flipped_padded, dim=0)

    batch_1 = Batch.from_data_list(prot_1s)
    batch_2 = Batch.from_data_list(prot_2s)
    
    # Return padded descriptors + length information
    return (batch_1, batch_2, labels, 
            p1_straight_padded, p1_flipped_padded, p2_straight_padded, p2_flipped_padded,
            p1_straight_lengths, p1_flipped_lengths, p2_straight_lengths, p2_flipped_lengths)

base_dir = '../masif_features'
processed_dir = os.path.join(base_dir, 'processed/')

# npy_file = os.path.join(base_dir, '/workspace/masif_features/fin_upd_test.npy')
# npy_file = os.path.join(base_dir, '/workspace/masif_features/full_data.npy')
train_npy_file = os.path.join(base_dir, 'trainset.npy')
test_npy_file = os.path.join(base_dir, 'valset.npy')
final_test_npy_file = os.path.join(base_dir, 'testset.npy')
# npy_file = os.path.join(base_dir, '/workspace/masif_features/test_mas.npy')
trainset = LabelledDataset(npy_file=train_npy_file, processed_dir=processed_dir)
testset = LabelledDataset(npy_file=test_npy_file, processed_dir=processed_dir)
final_testset = LabelledDataset(npy_file=final_test_npy_file, processed_dir=processed_dir)
# final_pairs = np.load(npy_file)
# size = final_pairs.shape[0]

# Using the same seed defined at the top of the file
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# indexes = list(range(size))
# random.shuffle(indexes, random.seed(SEED))

# train_ids = []
# test_ids = []

# for i in range(args.cv_fold):
#     if i == args.cv_fold_idx:
#         test_ids.extend(indexes[i::args.cv_fold])
#     else:
#         train_ids.extend(indexes[i::args.cv_fold])

# trainset = torch.utils.data.Subset(dataset, train_ids)
# testset = torch.utils.data.Subset(dataset, test_ids)

# trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size)])

# trainset = dataset
# npy_test_file = os.path.join(base_dir, 'fin_upd_test.npy')
# testset = LabelledDataset(npy_file=npy_test_file, processed_dir=processed_dir, desc=False)

trainloader = DataLoader(
    trainset,
    batch_size=1,
    num_workers=0,
    shuffle=True,
    collate_fn=collate_fn_unpadded,  # crucial
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)  # Set seed for dataloader workers
)

testloader = DataLoader(
    testset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    collate_fn=collate_fn_unpadded,  # crucial
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)  # Set seed for dataloader workers
)

final_testloader = DataLoader(
    final_testset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    collate_fn=collate_fn_unpadded,  # crucial
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)  # Set seed for dataloader workers
)
