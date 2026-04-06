import torch
import torch.nn as nn
import torch.nn.functional as F
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

class HyperPINO(nn.Module):
    def __init__(self, target_width=128, freq=10.0): # freq를 5.0으로 낮춰 고주파 노이즈 억제
        super().__init__()
        self.hyper_net_u = PointNetfeat(global_feat=True)
        self.hyper_net_mu = PointNetfeat(global_feat=True)
        
        self.register_buffer("FF", torch.randn(3, target_width // 2) * freq)
        encoded_dim = target_width 
        mlp_input_dim = encoded_dim + 4 

        # u 타겟 네트워크
        self.u_layers = nn.ModuleList([
            FCLayer_batch(mlp_input_dim, target_width),
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 3) 
        ])
        self.u_sizes = [l.get_param_size() for l in self.u_layers]
        self.param_proj_u = nn.Linear(1024, sum(self.u_sizes))

        # mu 타겟 네트워크
        self.mu_layers = nn.ModuleList([
            FCLayer_batch(mlp_input_dim, target_width),
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 1) 
        ])
        self.mu_sizes = [l.get_param_size() for l in self.mu_layers]
        self.param_proj_mu = nn.Linear(1024, sum(self.mu_sizes))

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            torch.nn.init.uniform_(self.param_proj_u.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.param_proj_mu.weight, -1e-3, 1e-3)
            # 초기 mu_pred가 약 1.2(간 평균값)가 되도록 바이어스 조정
            self.param_proj_mu.bias.data.fill_(0.0)

    def input_encoding(self, pos):
        x_proj = torch.matmul(pos, self.FF)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, pos, full_input):
        global_feat_u = self.hyper_net_u(full_input)
        global_feat_mu = self.hyper_net_mu(full_input)
        
        params_u_all = self.param_proj_u(global_feat_u)
        params_mu_all = self.param_proj_mu(global_feat_mu)
        
        encoded_pos = self.input_encoding(pos)
        tool_cond = full_input[:, :, 3:7]
        mlp_input = torch.cat([encoded_pos, tool_cond], dim=-1)

        # u 추론
        u_offsets = [0] + torch.cumsum(torch.tensor(self.u_sizes), dim=0).tolist()
        x_u = mlp_input
        for i in range(len(self.u_layers) - 1):
            p = params_u_all[:, u_offsets[i]:u_offsets[i+1]]
            x_u = torch.tanh(self.u_layers[i](x_u, p))
        u_pred = self.u_layers[-1](x_u, params_u_all[:, u_offsets[-2]:u_offsets[-1]])

        # mu 추론
        mu_offsets = [0] + torch.cumsum(torch.tensor(self.mu_sizes), dim=0).tolist()
        x_mu = mlp_input
        for i in range(len(self.mu_layers) - 1):
            p = params_mu_all[:, mu_offsets[i]:mu_offsets[i+1]]
            x_mu = torch.tanh(self.mu_layers[i](x_mu, p))
        mu_raw = self.mu_layers[-1](x_mu, params_mu_all[:, mu_offsets[-2]:mu_offsets[-1]])
        
        
       mu_pred = 0.3 + torch.sigmoid(mu_raw) * 1.5
        
        return u_pred, mu_pred