"""
H5Dataset                 : for sen1 or sen2 data determined by DataType Parameter
H5DatasetSelectIndexing   : select certain channel in raw sen1 or sen2 data
H5DatasetResample         : resample dataset following Indices
H5DatasetCat              : concanate sen1 and sen2 to generate 18 channels input
H5DatasetSia              : return two inputs for siamese network
"""
import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import h5py
import os
import torchvision.transforms as tr
import scipy.stats as ss
import scipy.misc as sm

class H5Dataset(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    Input Para:
        root(str): root where locate dataset
        mode(str): training or validation
        DataType(str): sen1 or sen2
    arguments:
        Data_tensor  (Tensor): contains sample sen1 data 8x32x32 or sen2 data 10x32x32
        Target_tensor(Tensor): label for corresponding indexing data
    return 
        Input(Tensor) for CNN 
        Anno(int)
    """
    def __init__(self, root, mode, DataType):
        self.root = root
        self.mode = mode
        self.data_type = DataType
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.Data_tensor = self.fid[self.data_type]
        self.Target_tensor = self.fid['label']

        self.mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_' + self.data_type +'.npy' )
        self.std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_' + self.data_type + '.npy' )

    def __getitem__(self, index):
        #Input = (self.Data_tensor[index] - self.mean)/self.std
        Input = (self.Data_tensor[index] - self.mean)/self.std
        Input = torch.from_numpy(Input).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input, Anno

    def __len__(self):
        return self.Target_tensor.shape[0]

class H5DatasetSelectIndexing(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    Input Para:
        root(str): root where locate dataset
        mode(str): training or validation
        DataType(str): sen1 or sen2
    arguments:
        Data_tensor  (Tensor): contains sample sen1 data 8x32x32 or sen2 data 10x32x32
        Target_tensor(Tensor): label for corresponding indexing data
    return 
        Input(Tensor) for CNN 
        Anno(int)
    """
    def __init__(self, root, mode, DataType):
        self.root = root
        self.mode = mode
        self.data_type = DataType
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.Data_tensor = self.fid[self.data_type]
        self.Target_tensor = self.fid['label']

        self.mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_' + self.data_type +'.npy' )
        self.std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_' + self.data_type + '.npy' )

    def __getitem__(self, index):
        #Input = (self.Data_tensor[index] - self.mean)/self.std
        Input = (self.Data_tensor[index] - self.mean)
        Input = torch.from_numpy(Input).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        Res10 = torch.index_select(dixhuittensor, 0, torch.LongTensor([0, 1, 2]))
        Res20 = torch.index_select(dixhuittensor, 0, torch.LongTensor([3, 4, 5, 7, 8, 9]))
        return Res10, Res20, Anno

    def __len__(self):
        return self.Target_tensor.shape[0]

class H5DatasetResample(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    Input Para:
        root(str): root where locate dataset
        mode(str): training or validation
        DataType(str): sen1 or sen2
    arguments:
        Data_tensor  (Tensor): contains sample sen1 data 8x32x32 or sen2 data 10x32x32
        Target_tensor(Tensor): label for corresponding indexing data
    return 
        Input(Tensor) for CNN 
        Anno(int)
    """
    def __init__(self, root, mode, DataType, indices):
        self.root = root
        self.mode = mode
        self.data_type = DataType
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.Data_tensor = self.fid[self.data_type]
        self.Target_tensor = self.fid['label']
        self.indices = indices

        self.mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_' + self.data_type +'.npy' )
        self.std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_' + self.data_type + '.npy' )

    def __getitem__(self, idx):
        #Input = (self.Data_tensor[index] - self.mean)
        index = self.indices[idx]
        Input = (self.Data_tensor[index] - self.mean)/self.std
        Input = torch.from_numpy(Input).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input, Anno

    def __len__(self):
        return len(self.indices)

class H5DatasetCat(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data
        labels      (Tensor): label for corresponding indexing data
    """    
    def __init__(self, sen1_tensor,sen2_tensor, Target_tensor):
        assert sen1_tensor.shape[0] == Target_tensor.shape[0]
        self.sen1_tensor = sen1_tensor
        self.sen2_tensor = sen2_tensor
        self.Target_tensor = Target_tensor

        self.sen1mean = np.array([-3.59122426e-05,-7.65856128e-06,5.93738575e-05,2.51662315e-05,4.42011066e-02,2.57610271e-01,7.55674337e-04,1.35034668e-03])
        self.sen1std = np.array([0.17555201, 0.17556463, 0.45998793, 0.45598876, 2.85599092, 8.32480061, 2.44987574, 1.4647353])

        self.sen2mean = np.array([0.12375696, 0.10927746, 0.10108552, 0.11423986, 0.15926567, 0.18147236, 0.17457403, 0.19501607, 0.15428469, 0.10905051])
        self.sen2std = np.array([0.03958796, 0.04777826, 0.06636617, 0.06358875, 0.07744387, 0.09101635, 0.09218467, 0.10164581, 0.09991773, 0.08780633])

    def __getitem__(self, index):
        Input_sen1  = (self.sen1_tensor - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor - self.sen2mean)/self.sen2std
        Input = np.concatenate((Input_sen1, Input_sen2), axis = 0)
        Input = torch.from_numpy(Input).permute(2,0,1).type(torch.FloatTensor)
        #Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        #Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #Input = torch.cat((Input_sen1, Input_sen2), 0)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input, Anno

    def __len__(self):
        return self.Target_tensor.shape[0]

class H5DatasetSiaResample(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode, indices, symmetrize = True):
        self.root = root
        self.mode = mode
        self.symmetrize = symmetrize
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)
        #if indices != None :
        #    self.indices = range(len(self.Target_tensor))
        #else :
        self.indices = indices

    def __getitem__(self, idx):
        index = self.indices[idx]
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        if self.symmetrize :
            if np.random.rand() < 0.5:
                Input_sen1, Input_sen2 = Input_sen1[ :, :: -1, :] - np.zeros_like(Input_sen1), Input_sen2[ :, :: -1, :] - np.zeros_like(Input_sen2)
            if np.random.rand() < 0.5:
                Input_sen1, Input_sen2 = Input_sen1[ :: -1, :, :] - np.zeros_like(Input_sen1), Input_sen2[ :: -1, :, :] - np.zeros_like(Input_sen2)


        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input_sen1, Input_sen2, Anno

    def __len__(self):
        return len(self.indices)#self.Target_tensor.shape[0]

class H5DatasetSia(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode, symmetrize = True):
        self.root = root
        self.mode = mode
        self.symmetrize = symmetrize
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)
        
    def __getitem__(self, index):
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        if self.symmetrize :
            if np.random.rand() < 0.5:
                Input_sen1, Input_sen2 = Input_sen1[ :, :: -1, :] - np.zeros_like(Input_sen1), Input_sen2[ :, :: -1, :] - np.zeros_like(Input_sen2)
            if np.random.rand() < 0.5:
                Input_sen1, Input_sen2 = Input_sen1[ :: -1, :, :] - np.zeros_like(Input_sen1), Input_sen2[ :: -1, :, :] - np.zeros_like(Input_sen2)

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input_sen1, Input_sen2, Anno

    def __len__(self):
        return self.Target_tensor.shape[0]

class H5DatasetTensorAnno(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode, indices):
        self.root = root
        self.mode = mode
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)
        self.indices = indices

    def __getitem__(self, idx):
        index = self.indices[idx]
        #index = idx 
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(2*self.Target_tensor[index]-1).float()
        return Input_sen1, Input_sen2, AnnoTensor

    def __len__(self):
        return len(self.indices)

class H5DatasetSoftAnno(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode, indices):
        self.root = root
        self.mode = mode
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.softlabel = np.load('data/soft_label/' + self.mode + '_soft_labels_851.npy')

        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)
        if indices == None :
            self.indices = range(len(self.Target_tensor))
        else :
            self.indices = indices
    def __getitem__(self, idx):
        index = self.indices[idx]
        #index = idx 
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        softlabel_tensor = torch.from_numpy(self.softlabel[index]).type(torch.FloatTensor)
        return Input_sen1, Input_sen2, Anno, softlabel_tensor

    def __len__(self):
        return len(self.indices)

class H5DatasetMerge(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)


    def __getitem__(self, index):
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)

        AnnoTensor = torch.from_numpy(self.Target_tensor[index])
        Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input_sen1, Input_sen2, sen2nonstd, Anno

    def __len__(self):
        return self.Target_tensor.shape[0]

class MultiLabel_H5DatasetSia(data.Dataset):
    """
    dataset warping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first dimenson
    
    arguments:
        sen1_tensor (Tensor): contains sample sen1 data 8x32x32
        sen2_tensor (Tensor): contains sample sen2 data 10x32x32
        labels      (Tensor): label for corresponding indexing data    
    """    
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.fid = h5py.File(self.root + '/' + self.mode + '.h5', 'r')
        self.sen1_tensor = self.fid['sen1']
        self.sen2_tensor = self.fid['sen2']
        self.Target_tensor = self.fid['label']
        self.sen1mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen1.npy' )
        self.sen1std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen1.npy' )
        self.sen2mean = np.load(self.root + '/mean_et_std/' + self.mode + '_mean_sen2.npy' )
        self.sen2std = np.load(self.root + '/mean_et_std/' + self.mode + '_std_sen2.npy' )
        self.mean = np.concatenate((self.sen1mean, self.sen2mean), axis = 0)
        self.std = np.concatenate((self.sen1std, self.sen2std), axis = 0)


    def __getitem__(self, index):
        Input_sen1  = (self.sen1_tensor[index] - self.sen1mean)/self.sen1std
        Input_sen2  = (self.sen2_tensor[index] - self.sen2mean)/self.sen2std

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        #sen2nonstd = torch.from_numpy((self.sen2_tensor[index] - self.sen2mean)).permute(2,0,1).type(torch.FloatTensor)
        
        AnnoList1 = np.nonzero(self.Target_tensor[index])[0][0]
        AnnoList2 = 18 if AnnoList1 > 9 else 17
        print(AnnoList1, AnnoList2)
        '''
        Anno = self.Target_tensor[index]
        AnnoMultiLabel = np.concatenate((Anno, [max()], ), axis = 0)'''
        #AnnoTensor = torch.nonzero(torch.from_numpy(AnnoMultiLabel) )
        #Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        return Input_sen1, Input_sen2, torch.LongTensor([AnnoList1, AnnoList2])

    def __len__(self):
        return self.Target_tensor.shape[0]

def Symmetrize(Input_sen1, Input_sen2):
    # reflect left right with 50% probability
    if np.random.rand() < 0.5:
        Input_sen1 = np.fliplr(Input_sen1)
        Input_sen2 = np.fliplr(Input_sen2)
    # reflect up down with 50% probability
    if np.random.rand() < 0.5:
        Input_sen1 = np.flipud(Input_sen1)
        Input_sen2 = np.flipud(Input_sen2)
    Input_sen1 = Input_sen1[:, : , ::-1] - np.zeros_like(Input_sen1)
    Input_sen2 = Input_sen2[:, : , ::-1] - np.zeros_like(Input_sen2)
    return Input_sen1, Input_sen2













































