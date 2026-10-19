# TripleMixer revision fcafe56f4f224a24ec566539b6731d9d2c700b1f; original module, no algorithm changes.
# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## Geometry Mixer
class GeoMixer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.channels_in, self.channels_out = channels_in, channels_out
        self.norm = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_out, 1)

        self.fc = nn.Conv2d(2 * channels_in, 2 * channels_in, (1, 1), bias=False)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(2 * channels_in),
            nn.Conv2d(2 * channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)


    def forward(self, x, neighbors):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""
        x = self.norm(x)
        point_emb = self.conv1(x)

        gather = []
        for ind_nn in range(
            1, neighbors.shape[1]
        ):
            temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, x.shape[1], -1)
            gather.append(torch.gather(x, 2, temp).unsqueeze(-1))

        neigh_fea = torch.cat(gather, -1)
        neigh_emb = neigh_fea - x.unsqueeze(-1)

        neigh_embfea = torch.cat([neigh_fea, neigh_emb], dim=1)

        att_activation = self.fc(neigh_embfea)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = neigh_embfea * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)

        finl_emb = self.conv2(f_agg).max(-1)[0]

        return self.final(torch.cat((point_emb, finl_emb), dim=1))
