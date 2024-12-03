import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader,Subset
import numpy as np

# TrafficDataset class for loading X, Y
class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        self.X = X  # Normalize the data between [0, 1]
        self.Y = Y  # Normalize the data between [0, 1]
        self.mean = 0  
        self.std = 1  

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

# Function to load data from HDF5 files and return DataLoaders
def load_data(batch_size, val_batch_size, data_root, num_workers):
    # List of data paths
    data_paths = [
        os.path.join(data_root, 'BJ13_M32x32_T30_InOut.h5'),
        os.path.join(data_root, 'BJ14_M32x32_T30_InOut.h5'),
        os.path.join(data_root, 'BJ15_M32x32_T30_InOut.h5'),
        os.path.join(data_root, 'BJ16_M32x32_T30_InOut.h5')
    ]

    # Initialize lists for X and Y to store the loaded data
    X_train_list, Y_train_list, X_test_list, Y_test_list = [], [], [], []

    X_list = []
    Y_list = []

    # Iterate over each file to load the data
    for data_path in data_paths:
        with h5py.File(data_path, 'r') as f:
            data = np.array(f['data'])

            X = data[:-1]  # Inputs (all but the last time step)
            Y = data[1:]   # Labels (all but the first time step)
            
            X_list.append(X)
            Y_list.append(Y)

    # Concatenate all years' data
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    split_idx = int(0.8 * X.shape[0])  # 80% train, 20% test
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]

    # Create dataset objects
    train_set = TrafficDataset(X=X_train, Y=Y_train)
    test_set = TrafficDataset(X=X_test, Y=Y_test)

    # Create DataLoaders
    train_set=Subset(train_set,range(96))
    test_set=Subset(test_set,range(96))
    dataloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    

    # Since we're not splitting validation data explicitly, set validation to None for now
    return dataloader_train, None, dataloader_test, 0, 1 

# import torch
# import numpy as np
# from torch.utils.data import Dataset


# class TrafficDataset(Dataset):
#     def __init__(self, X, Y):
#         super(TrafficDataset, self).__init__()
#         self.X = (X + 1) / 2
#         self.Y = (Y + 1) / 2
#         self.mean = 0
#         self.std = 1

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, index):
#         data = torch.tensor(self.X[index, ::]).float()
#         labels = torch.tensor(self.Y[index, ::]).float()
#         return data, labels

# def load_data(
#         batch_size, val_batch_size,
#         data_root, num_workers):

#     dataset = np.load(data_root+'taxibj/dataset.npz')
#     X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']

#     train_set = TrafficDataset(X=X_train, Y=Y_train)
#     test_set = TrafficDataset(X=X_test, Y=Y_test)

#     dataloader_train = torch.utils.data.DataLoader(
#         train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
#     dataloader_test = torch.utils.data.DataLoader(
#         test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

#     return dataloader_train, None, dataloader_test, 0, 1