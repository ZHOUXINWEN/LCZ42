"""
Neural Network Factory

"""

from densenet import DenseNet, DenseNetSia
from nasnet_mobile import nasnetamobile
from senet_shallow import se_resnet50_shallow
from resnext import CifarResNeXt, ShallowResNeXt
from senet_shallow_sia import se_resnet50_shallow_sia, se_resnext50_32x4d
from SimpleNet import SimpleNet, SimpleNetLeaky, SimpleNet4x4,SimpleNetSen2, SimpleNetGN
from shake_shake import Shake_Shake

class NetworkFactory:
    """ Factory Class for Neural Network """
    @staticmethod
    def ConsturctNetwork(NetworkType, resume):
        """ Creates a Neural Network of the specified type.

        Parameters
        ----------
        NetworkType : :obj:`str`
            the type of Neural Network 
        resume : :obj:'str'
            the path of trained model
        """
        if NetworkType == 'CifarResNeXt' :
            model = CifarResNeXt(num_classes = 17, depth = 29, cardinality = 8)    

        elif NetworkType == 'ShallowResNeXt' :
            model = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16)    

        elif NetworkType == 'se_resnet50_shallow_sia' :
            model = se_resnet50_shallow_sia(17, None)    

        elif NetworkType == 'se_resnext50_32x4d' :
            model = se_resnet50_shallow_sia(17, None)
        elif NetworkType == 'SimpleNet':
            model = SimpleNet(17)    

        elif NetworkType == 'SimpleNetGN':
            model = SimpleNetGN(17)    

        elif NetworkType == 'DenseNet':
            model = DenseNet(num_classes = 17)    

        elif NetworkType == 'DenseNetSia':
            model = DenseNetSia(num_classes = 17)    

        elif NetworkType == 'nasnetamobile':
            model = nasnetamobile(num_classes = 17, pretrained = None)

        elif NetworkType == 'Shake_Shake' :   
            model = Shake_Shake(num_classes = 17)

        else:
            raise ValueError('Neural Network type %s not supported' %(NetworkType))

        if resume:
            model.load_state_dict(torch.load(args.resume))

        return model
