'''
train Random Forest From merged models' normalise digits

'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def TrainRF_fromSoftLabel(SoftLabel_path, Anno_path):
    '''
    SoftLabel_path:  path for feature for training
    '''
    SoftLabel = np.load( SoftLabel_path)
    anno = np.load( Anno_path)

    #initialise Random Forest and train
    clf = RandomForestClassifier(n_estimators = 100, max_features = 'log2')
    clf.fit(SoftLabel, anno)

    joblib.dump(clf, '100_estimator_max_features_log2_RandomForest.pkl')
    return clf


def TestRF(SoftLabel_path, Anno_path, clf):

    SoftLabel = np.load( SoftLabel_path)
    anno = np.load( Anno_path)

    pred_result = clf.predict(SoftLabel)
    cnf_matrix = confusion_matrix(anno, pred_result)
    np.save('851_RF_training_result.npy', pred_result)
    np.save('confmat_RF_max_features_log2.npy',cnf_matrix)
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')
    print(cnf_tr/len(anno))


if __name__ == '__main__':
    
    SoftLabel_path = 'data/soft_label/training_soft_labels_851.npy'
    Anno_path = 'data/training_int_anno.npy'
    clf = TrainRF_fromSoftLabel(SoftLabel_path, Anno_path)
    '''
    clf = RandomForestClassifier(n_estimators = 100, max_features = 'log2')
    clf = joblib.load('100_estimator_max_features_log2_RandomForest.pkl')
    '''
    validation_path = 'data/soft_label/validation_soft_labels_851.npy'
    validation_anno_path = 'data/validation_int_anno.npy'
    TestRF(validation_path , validation_anno_path, clf)
