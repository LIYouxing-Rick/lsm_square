from src.network import *
import torch
import torch.nn as nn
import warnings
__all__ = ['Newmodel', 'get_model']

class HyperbolicClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(HyperbolicClassifier, self).__init__()
        import geoopt
        self.manifold = geoopt.manifolds.PoincareBall(c=1.0)
        self.weight_v = geoopt.ManifoldParameter(self.manifold.random((in_dim, num_classes)), manifold=self.manifold)
        self.weight_g = nn.Parameter(torch.ones(num_classes))
        self.gamma = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        from correlation_metric.ppbcm_functionals import unidirectional_poincare_mlr
        logits = unidirectional_poincare_mlr(x, self.weight_g, self.weight_v, self.gamma, self.manifold.c)
        return logits

class Newmodel(Basemodel):
    """replace the image representation method and classifier

       Args:
       modeltype: model archtecture
       representation: image representation method
       num_classes: the number of classes
       freezed_layer: the end of freezed layers in network
       pretrained: whether use pretrained weights or not
    """
    def __init__(self, modeltype, representation, num_classes, freezed_layer, pretrained=False):
        super(Newmodel, self).__init__(modeltype, pretrained)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            representation_args['input_dim'] = self.representation_dim
            self.representation = representation_method(**representation_args)
            fc_input_dim = self.representation.output_dim
            if not pretrained:
                if isinstance(self.classifier, nn.Sequential): # for alexnet and vgg*
                    conv6_index = 0
                    for m in self.classifier.children():
                        if isinstance(m, nn.Linear):
                            output_dim = m.weight.size(0) # 4096
                            self.classifier[conv6_index] = nn.Linear(fc_input_dim, output_dim)
                            break
                        conv6_index += 1
                    if representation_args.get('corr_method', None) == 'phcm':
                        self.classifier[-1] = HyperbolicClassifier(output_dim, num_classes)
                    else:
                        self.classifier[-1] = nn.Linear(output_dim, num_classes)
                else:
                    if representation_args.get('corr_method', None) == 'phcm':
                        self.classifier = HyperbolicClassifier(fc_input_dim, num_classes)
                    else:
                        self.classifier = nn.Linear(fc_input_dim, num_classes)
            else:
                if representation_args.get('corr_method', None) == 'phcm':
                    self.classifier = HyperbolicClassifier(fc_input_dim, num_classes)
                else:
                    self.classifier = nn.Linear(fc_input_dim, num_classes)
        else:
            if modeltype.startswith('alexnet') or modeltype.startswith('vgg'):
                output_dim = self.classifier[-1].weight.size(1) # 4096
                self.classifier[-1] = nn.Linear(output_dim, num_classes)
            else:
                self.classifier = nn.Linear(self.representation_dim, num_classes)
        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    m = self._freeze(m)
                index_before_freezed_layer += 1

    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


def get_model(modeltype, representation, num_classes, freezed_layer, pretrained=False):
    _model = Newmodel(modeltype, representation, num_classes, freezed_layer, pretrained=pretrained)
    return _model
