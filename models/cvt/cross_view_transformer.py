import torch
import torch.nn as nn
import numpy as np

from models.cvt.decoder import *
from models.cvt.encoder import *
from models.gpn.density import Density, Evidence


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4
    ):
        super().__init__()
        print("Initializing CVT model")

        self.encoder = Encoder()
        self.decoder = Decoder(128, [128, 128, 64])

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, outC, 1))

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        batch = {
            'image': imgs,
            'intrinsics': self.convert(intrins),
            'extrinsics': extrins
        }
        x = self.encoder(batch)
        y = self.decoder(x)
        z = self.to_logits(y)

        return z

    def convert(self, intrins):
        intrins[:, :, 0, 0] *= W/1600
        intrins[:, :, 0, 2] *= W/1600
        intrins[:, :, 1, 1] *= (H+O)/900
        intrins[:, :, 1, 2] *= (H+O)/900
        intrins[:, :, 1, 2] -= O
        return intrins


class CrossViewTransformerDropout(CrossViewTransformer):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4
    ):
        super(CrossViewTransformerDropout, self).__init__(outC=outC, dim_last=dim_last)

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
            nn.Conv2d(dim_last, outC, 1)
        )

        self.tests = -1

        self.decoder.dropout = True

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        if self.tests > 0:
            outputs = []

            for i in range(self.tests):
                outputs.append(super().forward(x, rots, trans, intrins, post_rots, post_trans))

            return torch.mean(torch.stack(outputs), dim=0)
        else:
            return super().forward(x, rots, trans, intrins, extrins, post_rots, post_trans)


class CrossViewTransformerEnsemble(CrossViewTransformer):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4
    ):
        super(CrossViewTransformerEnsemble, self).__init__(outC=outC, dim_last=dim_last)

        num_models = 5
        self.models = nn.ModuleList([CrossViewTransformer(outC=outC) for _ in range(num_models)])

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []

        for model in self.models:
            outputs.append(model(x, rots, trans, intrins, extrins, post_rots, post_trans))

        return torch.mean(torch.stack(outputs), dim=0)


class CrossViewTransformerGPN(CrossViewTransformer):
    def __init__(
            self,
            dim_last: int = 64,
            outC: int = 4
    ):
        super(CrossViewTransformerGPN, self).__init__(outC=outC, dim_last=dim_last)

        self.outC = outC
        self.latent_size = 16
        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=outC)
        self.evidence = Evidence(scale='latent-new')

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, self.latent_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Conv2d(outC, outC, 1)
        self.p_c = None

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        batch = {
            'image': imgs,
            'intrinsics': self.convert(intrins),
            'extrinsics': extrins
        }

        x = self.encoder(batch)
        x = self.decoder(x)
        x = self.to_logits(x)

        x = x.permute(0, 2, 3, 1).to(x.device)
        x = x.reshape(-1, self.latent_size)

        self.p_c = self.p_c.to(x.device)

        log_q_ft_per_class = self.flow(x) + self.p_c.view(1, -1).log()

        beta = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=2.0).exp()

        beta = beta.reshape(-1, 200, 200, self.outC).permute(0, 3, 1, 2).contiguous()

        if self.last is not None:
            beta = self.last(beta.log()).exp()
        alpha = beta + 1

        return alpha.clamp(min=1e-4)
