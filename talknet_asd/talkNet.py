import torch
import torch.nn as nn

import sys, time

from talknet_asd.loss import lossAV, lossA, lossV
from talknet_asd.model.talkNetModel import talkNetModel


class talkNet(nn.Module):
    def __init__(self, device="auto", dtype=torch.bfloat16, **kwargs):
        super(talkNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.model = talkNetModel()
        self.lossAV = lossAV().to(self.device, dtype=dtype)
        print(
            time.strftime("%m-%d %H:%M:%S")
            + " Model para number = %.2f"
            % (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024)
        )

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=self.device)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origName, selfState[name].size(), loadedState[origName].size())
                )
                continue
            selfState[name].copy_(param)
        self.to(self.device, dtype=self.dtype)
        self.model.eval()
