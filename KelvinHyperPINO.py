import torch
import torch.nn as nn
from pointnet import PointNetfeat

class FCLayer_batch(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
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

        self.size1 = self.layer1.get_param_size()
        self.size2 = self.layer2.get_param_size()
        self.size3 = self.layer3.get_param_size()
        self.size4 = self.layer4.get_param_size()
        
        total_size = self.size1 + self.size2 + self.size3 + self.size4
        self.param_proj = nn.Linear(1024, total_size)

        # 🚀 [수정] 초기화 강도 상향: 평면의 늪 탈출용 (1e-4 -> 1e-2)
        with torch.no_grad():
            torch.nn.init.uniform_(self.param_proj.weight, -1e-2, 1e-2)
            torch.nn.init.constant_(self.param_proj.bias, 0.0)

    def forward(self, pos, full_input):
        global_feat = self.hyper_net(full_input)
        all_params = self.param_proj(global_feat)
        
        p1 = all_params[:, :self.size1]
        p2 = all_params[:, self.size1 : self.size1+self.size2]
        p3 = all_params[:, self.size1+self.size2 : self.size1+self.size2+self.size3]
        p4 = all_params[:, -self.size4:]

        x = torch.tanh(self.layer1(pos, p1))
        x = torch.tanh(self.layer2(x, p2))
        x = torch.tanh(self.layer3(x, p3))
        out = self.layer4(x, p4)

        u_pred = out[:, :, 0:3] 
        mu_raw = out[:, :, 3:4]
        
        # 유니버셜 스케일 (0.01 ~ 10.01 kPa)
        mu_pred = 0.01 + torch.sigmoid(mu_raw) * 10.0 

        return u_pred, mu_pred