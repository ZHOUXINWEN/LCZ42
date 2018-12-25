import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def PlotMixtrainingAccCurve(model_dir):
    Accuracy_lists = []
    for epoch in range(16):
        if epoch%2 == 0:
            for itera in range(1, 4):
                iteration = itera*2447-1
                cnf_matrix = np.load(model_dir + '/LCZ42_SGD_' + repr(epoch) + '_' +repr(iteration) + '.npy')
                lens = np.sum(cnf_matrix)
                cnf_tr = np.trace(cnf_matrix)
                Accuracy = cnf_tr.astype('float')/lens
                Accuracy_lists.append(Accuracy)
        else :
            pass
            '''
            for itera in range(1,2):
                iteration = itera*754-1
                cnf_matrix = np.load(model_dir + '/LCZ42_SGD_' + repr(epoch) + '_' +repr(iteration) + '.npy')
                lens = np.sum(cnf_matrix)
                cnf_tr = np.trace(cnf_matrix)
                Accuracy = cnf_tr.astype('float')/lens
                Accuracy_lists.append(Accuracy)
            '''
#print(len(Accuracy_lists))
#print(Accuracy_lists)
    plt.figure(figsize=(24, 24))
    print(Accuracy_lists)
    train_iters  = range(len(Accuracy_lists))
    for i in train_iters :
        plt.text(i, Accuracy_lists[i], repr(round(Accuracy_lists[i], 3) ),
                 color = 'black',
                 fontdict={'weight': 'normal', 'size': 18})

    plt.plot(train_iters, Accuracy_lists, linewidth=4, color='b')

    plt.ylim(0.65, 0.95, 0.05)
    plt.legend(('Accuracy'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.grid()
    plt.savefig(os.path.join(model_dir, 'training_curve.jpg'))


if __name__ == '__main__':
    model_dir = 'weights/CEL_symme_Tiers12_bs32_8cat10channel_SimpleNetGN'
    PlotMixtrainingAccCurve(model_dir)
