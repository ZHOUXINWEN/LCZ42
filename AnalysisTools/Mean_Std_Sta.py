import h5py
import numpy as np
mode = 'round1_test_a_20181109'
File_name = ['data/' + mode + '.h5']

Sam_idx = 0
Mean_8_cha = np.zeros((8), np.float64)
Mean_10_cha = np.zeros((10), np.float64)
for filename in File_name:
    File = h5py.File(filename, 'r')
    Data_8 = File['sen1']
    Data_10 = File['sen2']
    for i in range(Data_8.shape[0]):
        Mean_8_cha = (Mean_8_cha * Sam_idx + Data_8[i].mean(axis = (0, 1))) / (Sam_idx + 1) 
        Mean_10_cha = (Mean_10_cha * Sam_idx + Data_10[i].mean(axis = (0, 1))) / (Sam_idx + 1) 
        Sam_idx += 1
print "Mean of 8 channels:"
print Mean_8_cha
np.save('data/'+ mode +'_mean_sen1.npy', Mean_8_cha)
print "Mean of 10 channels:"
print Mean_10_cha
np.save('data/'+ mode +'_mean_sen2.npy', Mean_10_cha)

Sam_idx = 0
Std_8_cha = np.zeros((8), np.float64)
Std_10_cha = np.zeros((10), np.float64)
for filename in File_name:
    File = h5py.File(filename, 'r')
    Data_8 = File['sen1']
    Data_10 = File['sen2']
    for i in range(Data_8.shape[0]):
        Data_8_std = (Data_8[i] - Mean_8_cha) ** 2
        Data_10_std = (Data_10[i] - Mean_10_cha) ** 2
        Std_8_cha = (Std_8_cha * Sam_idx + Data_8_std.mean(axis = (0, 1))) / (Sam_idx + 1) 
        Std_10_cha = (Std_10_cha * Sam_idx + Data_10_std.mean(axis = (0, 1))) / (Sam_idx + 1) 
        Sam_idx += 1
print "Std of 8 channels:"
print np.sqrt(Std_8_cha)
np.save('data/'+ mode +'_std_sen1.npy', np.sqrt(Std_8_cha))
print "Std of 10 channels:"
print np.sqrt(Std_10_cha)
np.save('data/'+ mode +'_std_sen2.npy', np.sqrt(Std_10_cha))    
