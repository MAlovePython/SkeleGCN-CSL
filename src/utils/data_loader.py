import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.nn.functional as F

class CSL500Dataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        if self.augment:
            sample = self.augment_data(sample)

        pose = torch.from_numpy(sample.copy()).float()
        label = torch.tensor(self.labels[idx]).long()
        return pose, label

    def augment_data(self, sample):
        if np.random.rand() < 0.5:
            sample = sample + np.random.normal(0, 0.01, sample.shape)
        if np.random.rand() < 0.5:
            sample = np.flip(sample, axis=0).copy()  # 时间翻转并复制
        return sample

def collate_fn(batch):
    poses, labels = zip(*batch)

    # 获取每个序列的长度
    lengths = torch.tensor([p.shape[0] for p in poses])
    max_len = lengths.max().item()

    # 填充序列到最大长度
    padded_poses = []
    for pose in poses:
        pad_length = max_len - pose.shape[0]
        padded_pose = F.pad(pose, (0, 0, 0, pad_length))
        padded_poses.append(padded_pose)

    # 堆叠填充后的序列
    poses_padded = torch.stack(padded_poses)

    # 转换标签为张量
    labels = torch.tensor(labels, dtype=torch.long)

    return poses_padded, labels, lengths

class KFoldDataLoader:
    def __init__(self, data_path, labels_path, batch_size, n_splits=5, shuffle=True):
        self.data = np.load(data_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True)
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=self.n_splits, shuffle=shuffle)

    def get_loaders(self):
        for train_idx, val_idx in self.kfold.split(self.data):
            train_data, train_labels = self.data[train_idx], self.labels[train_idx]
            val_data, val_labels = self.data[val_idx], self.labels[val_idx]

            train_dataset = CSL500Dataset(train_data, train_labels, augment=True)
            val_dataset = CSL500Dataset(val_data, val_labels, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

            yield train_loader, val_loader