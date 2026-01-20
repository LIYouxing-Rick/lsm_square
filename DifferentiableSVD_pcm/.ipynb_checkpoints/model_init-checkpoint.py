from src.network import *
import torch
import torch.nn as nn
import warnings
__all__ = ['Newmodel', 'get_model']

import torch
import torch.nn as nn
import geoopt


class MultiPrototypeHyperbolicClassifier(nn.Module):
    """
    Multi-prototype Hyperbolic RMLR Classifier

    - 每个类别 K 个超曲中心（prototype）
    - 使用 log-sum-exp 聚合
    - 与原 HyperbolicClassifier forward 接口完全一致
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_prototypes: int = 4,   # 👈 K
        init_gamma: float = 1.0,
        init_g: float = 1.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # Poincaré ball
        self.manifold = geoopt.manifolds.PoincareBall(c=1.0)

        # ---- 多 prototype 超曲中心 ----
        # shape: [C, K, D]
        self.weight_v = geoopt.ManifoldParameter(
            self.manifold.random((num_classes, num_prototypes, in_dim)),
            manifold=self.manifold
        )

        # 每个类别一个 scale（与原实现一致）
        self.weight_g = nn.Parameter(
            torch.ones(num_classes) * init_g
        )

        # margin / radius
        self.gamma = nn.Parameter(
            torch.tensor(init_gamma)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]   (Poincaré ball)
        return: [B, num_classes]
        """
        # ---- 曲率 ----
        c = torch.as_tensor(
            self.manifold.c,
            device=x.device,
            dtype=x.dtype
        )
        rc = torch.sqrt(c)

        # ---- prototypes ----
        # z: [C, K, D]
        z = self.weight_v

        # 正确的方向归一化（最后一维）
        z_norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        z_unit = z / z_norm

        # ---- 输入 ----
        rcx = rc * x                           # [B, D]
        cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)  # [B, 1]

        # ---- inner product ----
        # [B, D] · [C, K, D] → [B, C, K]
        inner = torch.einsum("bd,ckd->bck", rcx, z_unit)

        # ---- RMLR ----
        drcr = 2.0 * rc * self.gamma

        num = (
            2.0 * inner * torch.cosh(drcr)
            - (1.0 + cx2.unsqueeze(-1)) * torch.sinh(drcr)
        )
        den = torch.clamp(1.0 - cx2, min=1e-15)

        logits_ck = (
            2.0 * self.weight_g.view(1, -1, 1) / rc
            * torch.asinh(num / den.unsqueeze(-1))
        )  # [B, C, K]

        # ---- prototype 聚合（关键一步）----
        # 推荐：log-sum-exp（稳定 + 表达力强）
        logits = torch.logsumexp(logits_ck, dim=-1)  # [B, C]

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
                        self.classifier = MultiPrototypeHyperbolicClassifier(
    fc_input_dim,
    num_classes,
    num_prototypes=2   # 👈 建议 2 / 4 / 8 试
)

                    else:
                        self.classifier = nn.Linear(fc_input_dim, num_classes)
            else:
                if representation_args.get('corr_method', None) == 'phcm':
                   self.classifier = MultiPrototypeHyperbolicClassifier(
    fc_input_dim,
    num_classes,
    num_prototypes=2   # 👈 建议 2 / 4 / 8 试
)

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
