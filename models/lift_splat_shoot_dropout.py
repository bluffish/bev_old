from models.lift_splat_shoot import *


class LiftSplatShootDropout(LiftSplatShoot):
    def __init__(self, outC=4):
        super(LiftSplatShootDropout, self).__init__(outC=outC)

        self.bevencode.up1.conv = nn.Sequential(
            nn.Conv2d(self.bevencode.up1.in_channels, self.bevencode.up1.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bevencode.up1.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bevencode.up1.out_channels, self.bevencode.up1.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bevencode.up1.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
        )

        self.bevencode.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
            nn.Conv2d(128, self.outC, kernel_size=1, padding=0),
        )

        self.tests = -1

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):

        if self.tests > 0:
            outputs = []

            for i in range(self.tests):
                outputs.append(super().forward(x, rots, trans, intrins, post_rots, post_trans))

            return torch.mean(torch.stack(outputs), dim=0)
        else:
            return super().forward(x, rots, trans, intrins, post_rots, post_trans)