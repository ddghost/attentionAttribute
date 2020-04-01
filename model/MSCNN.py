
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import model.resnet as resnet

__all__ = ['MSCNN']
EPSILON = 1e-12
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class AdaptiveStdPool(nn.Module):
    def __init__(self):
        super(AdaptiveStdPool, self).__init__()

    def forward(self, x):
        avgPool = nn.AdaptiveAvgPool2d(1)
        avgNum = avgPool(x)
        stdMat = x - avgNum
        stdMat = stdMat * stdMat
        out = avgPool(stdMat)
        out = torch.sqrt( out + EPSILON )

        return out

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1) * 100
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class MSCNN(nn.Module):
    def __init__(self, num_classes, M=32, net='resnet50', pretrained=False, layerNum=4, lastLayerBlockNum=-1,useMultiScale=True):
        super(MSCNN, self).__init__()

        self.M = M
        self.net = net
        self.layerNum = layerNum
        self.num_classes = num_classes
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        #self.stdPool = AdaptiveStdPool()
        # Network Initialization
        self.useMultiScale = useMultiScale

        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'resnet' in net:
            #!!!!
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features(layerNum=layerNum, lastLayerBlockNum=lastLayerBlockNum)
            layerFeatureNum = 32
            self.num_features = 0
            if(self.useMultiScale):
                mutilMatrixLayerStart = 3
            else:
                mutilMatrixLayerStart = 4

            for i in range(1, layerNum+1):
                layerFeatureNum *= 2 
                if( i >= mutilMatrixLayerStart):
                    self.num_features += layerFeatureNum
            layerFeatureNum *= self.features[-1][-1].expansion
            self.num_features *= self.features[-1][-1].expansion
  
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        if(self.useMultiScale):
            #self.attentions2 = BasicConv2d(layerFeatureNum // 4, self.M, kernel_size=1)
            self.attentions3 = BasicConv2d(layerFeatureNum // 2, self.M, kernel_size=1)
            self.attentions4 = BasicConv2d(layerFeatureNum , self.M, kernel_size=1)
        else:
            self.attentions = BasicConv2d(layerFeatureNum, self.M, kernel_size=1)

        self.bap = BAP(pool='GAP')

        '''
        self.fc0 = nn.Linear(self.M * self.num_features // 4, classSetting[0], bias=False)
        self.fc1 = nn.Linear(self.M * self.num_features // 4, classSetting[1], bias=False)
        self.fc2 = nn.Linear(self.M * self.num_features // 4, classSetting[2], bias=False)
        self.fc3 = nn.Linear(self.M * self.num_features // 4, classSetting[3], bias=False)
        '''
        self.fc = nn.Linear(self.M * self.num_features, num_classes, bias=False)


        logging.info('MSCNN: using {} as feature extractor, classSetting: {}, num_attentions: {}'.format(net, self.classSetting, self.M))

        

    def forward(self, x):
        if(self.M == 1):
            return self.fcForward(x)
        else:
            return self.matrixForward(x)

    def fcForward(self, x):
        batch_size = x.size(0)
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu1(x)
        x = self.features.maxpool1(x)

        if(hasattr(self.features, 'layer1')):
            feature_maps1 = self.features.layer1(x)
        if(hasattr(self.features, 'layer2')):
            feature_maps2 = self.features.layer2(feature_maps1)
        if(hasattr(self.features, 'layer3')):
            feature_maps3 = self.features.layer3(feature_maps2)
        if(hasattr(self.features, 'layer4')):
            feature_maps4 = self.features.layer4(feature_maps3)

        if(self.useMultiScale):
            feature_maps3 = self.avgPool(feature_maps3).view(batch_size,-1)
            feature_maps4 = self.avgPool(feature_maps4).view(batch_size,-1)
            feature_maps = torch.cat((feature_maps3, feature_maps4), 1)
        else:
            feature_maps = self.avgPool(feature_maps).view(batch_size,-1)
        # Classification
        predict = []

        predict.append( self.fc0 (feature_maps) )
        predict.append( self.fc1 (feature_maps) )
        predict.append( self.fc2 (feature_maps) )
        predict.append( self.fc3 (feature_maps) )
        return predict

    '''
    def resetLinear(self):
        self.fc0 = nn.Linear(self.M * self.num_features, classSetting[0], bias=False)
        self.fc1 = nn.Linear(self.M * self.num_features, classSetting[1], bias=False)
        self.fc2 = nn.Linear(self.M * self.num_features, classSetting[2], bias=False)
        self.fc3 = nn.Linear(self.M * self.num_features, classSetting[3], bias=False)
        self.frezzeFromShallowToDeep(-1)
    '''

    def matrixForward(self, x):
        batch_size = x.size(0)
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu1(x)
        x = self.features.maxpool1(x)

        if(hasattr(self.features, 'layer1')):
            feature_maps1 = self.features.layer1(x)
        if(hasattr(self.features, 'layer2')):
            feature_maps2 = self.features.layer2(feature_maps1)
        if(hasattr(self.features, 'layer3')):
            feature_maps3 = self.features.layer3(feature_maps2)
        if(hasattr(self.features, 'layer4')):
            feature_maps4 = self.features.layer4(feature_maps3)


        if self.net != 'inception_mixed_7c':
            if(self.useMultiScale):
                #attention_maps2 = self.attentions2(feature_maps2)
                attention_maps3 = self.attentions3(feature_maps3)
                attention_maps4 = self.attentions4(feature_maps4)
            else:
                attention_maps4 = self.attentions(feature_maps4)
        else:
            pass
            #attention_maps = feature_maps[:, :self.M, ...]
        if(self.useMultiScale):
            #feature_matrix2 = self.bap(feature_maps2, attention_maps2)
            feature_matrix3 = self.bap(feature_maps3, attention_maps3)
            feature_matrix4 = self.bap(feature_maps4, attention_maps4)
            feature_matrix = torch.cat((feature_matrix3,feature_matrix4), 1)
        else:
            feature_matrix = self.bap(feature_maps4, attention_maps4)




        # Classification
        predict = self.fc (feature_matrix)

        return predict
    
    def frezzeFromShallowToDeep(self, lastLayer):
        #conv1 0, layer1 1, layer2 2, layer3 3, layer4 4
        for i, para in enumerate(self.features.parameters()):
            para.requires_grad = True
        if(lastLayer >= 0):
             for i, para in enumerate(self.features.conv1.parameters() ):
                para.requires_grad = False
             for i, para in enumerate(self.features.bn1.parameters() ):
                para.requires_grad = False
        if(lastLayer >= 1):
             for i, para in enumerate(self.features.layer1.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 2):
             for i, para in enumerate(self.features.layer2.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 3):
             for i, para in enumerate(self.features.layer3.parameters() ):
                para.requires_grad = False
        if(lastLayer >= 4):
             for i, para in enumerate(self.features.layer4.parameters() ):
                para.requires_grad = False
                
    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_Mutil, self).load_state_dict(model_dict)

    