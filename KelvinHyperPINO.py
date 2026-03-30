import torch
import torch.nn as nn
from pointnet import PointNetfeat

class FCLayer_batch(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in, self.num_out = num_in, num_out
        self.weight_size = num_out * num_in
        self.bias_size = num_out

    def forward(self, x, param):
        B = x.shape[0]
        w = param[:, : self.weight_size].reshape(B, self.num_out, self.num_in)
        b = param[:, self.weight_size :].reshape(B, 1, self.num_out)
        return torch.bmm(x, w.transpose(1, 2)) + b

    def get_param_size(self):
        return self.weight_size + self.bias_size

class KelvinHyperPINO(nn.Module):
    def __init__(self, target_width=128):
        super().__init__()
        self.hyper_net = PointNetfeat(global_feat=True)
        self.layer1 = FCLayer_batch(3, target_width)
        self.layer2 = FCLayer_batch(target_width, target_width)
        self.layer3 = FCLayer_batch(target_width, target_width)
        self.layer4 = FCLayer_batch(target_width, 4)

        self.sizes = [self.layer1.get_param_size(), self.layer2.get_param_size(),
                      self.layer3.get_param_size(), self.layer4.get_param_size()]
        
        self.param_proj = nn.Linear(1024, sum(self.sizes))

        # 🚀 참교육 1: 초기 곡률 부여 (1e-4 -> 1e-2)
        with torch.no_grad():
            torch.nn.init.uniform_(self.param_proj.weight, -1e-2, 1e-2)
            torch.nn.init.constant_(self.param_proj.bias, 0.0)
            self.param_proj.bias.data[-1] = -2.63 # logit(0.07) ≈ -2.63, 초기 mu_pred가 약 0.07이 되도록 설정

    def forward(self, pos, full_input):
        global_feat = self.hyper_net(full_input)
        all_params = self.param_proj(global_feat)
        
        offsets = [0] + torch.cumsum(torch.tensor(self.sizes), dim=0).tolist()
        params = [all_params[:, offsets[i]:offsets[i+1]] for i in range(4)]

        x = torch.tanh(self.layer1(pos, params[0]))
        x = torch.tanh(self.layer2(x, params[1]))
        x = torch.tanh(self.layer3(x, params[2]))
        out = self.layer4(x, params[3])

        u_pred = out[:, :, 0:3] 
        mu_raw = out[:, :, 3:4]
        
        # 🚀 참교육 2: 하한선 상향 (0.01 -> 0.03) 논문 근막 수치 반영
        mu_pred = 0.03 + torch.sigmoid(mu_raw) * 10.0 
        return u_pred, mu_pred