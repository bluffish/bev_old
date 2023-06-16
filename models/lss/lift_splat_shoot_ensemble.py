from models.lss.lift_splat_shoot import *


class LiftSplatShootEnsemble(nn.Module):
    def __init__(self, outC=4):
        super(LiftSplatShootEnsemble, self).__init__()

        num_models = 5
        self.models = nn.ModuleList([LiftSplatShoot(outC=outC) for _ in range(num_models)])

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []

        for model in self.models:
            outputs.append(model(x, rots, trans, intrins, extrins, post_rots, post_trans))

        return torch.stack(outputs, dim=0)