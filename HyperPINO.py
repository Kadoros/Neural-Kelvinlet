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
    def __init__(self, target_width=128, freq=10.0):
        super().__init__()
        
        # 1. 독립적인 피처 추출기
        self.hyper_net_u = PointNetfeat(global_feat=True)
        self.hyper_net_mu = PointNetfeat(global_feat=True)
        
        # 2. 푸리에 인코딩 설정
        self.register_buffer("FF", torch.randn(3, target_width // 2) * freq)
        encoded_dim = target_width 
        
        # 🌟 변경점: Target MLP의 입력 차원을 (푸리에 인코딩 128) + (Tool 정보 4) 로 늘림
        mlp_input_dim = encoded_dim + 4  # 132 차원

        # 3. 변위(u) 전용 타겟 네트워크
        self.u_layers = nn.ModuleList([
            FCLayer_batch(mlp_input_dim, target_width), # <-- 입력 차원 변경됨
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 3) 
        ])
        self.u_sizes = [l.get_param_size() for l in self.u_layers]
        self.param_proj_u = nn.Linear(1024, sum(self.u_sizes))

        # 4. 탄성도(mu) 전용 타겟 네트워크
        self.mu_layers = nn.ModuleList([
            FCLayer_batch(mlp_input_dim, target_width), # <-- 입력 차원 변경됨
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 1) 
        ])
        self.mu_sizes = [l.get_param_size() for l in self.mu_layers]
        self.param_proj_mu = nn.Linear(1024, sum(self.mu_sizes))

        # 5. 초기화 전략
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            torch.nn.init.uniform_(self.param_proj_u.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.param_proj_mu.weight, -1e-3, 1e-3)
            self.param_proj_mu.bias.data[-1] = -2.63

    def input_encoding(self, pos):
        """좌표를 고주파 영역으로 맵핑하는 푸리에 인코딩"""
        x_proj = torch.matmul(pos, self.FF)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, pos, full_input):
        # [Step 1~2] 하이퍼네트워크를 통해 가중치 생성 (기존 동일)
        global_feat_u = self.hyper_net_u(full_input)
        global_feat_mu = self.hyper_net_mu(full_input)
        
        params_u_all = self.param_proj_u(global_feat_u)
        params_mu_all = self.param_proj_mu(global_feat_mu)
        
        # [Step 3] 입력 인코딩 및 🌟 Tool 정보 결합
        encoded_pos = self.input_encoding(pos)                 # 형태: [B, N, 128]
        tool_cond = full_input[:, :, 3:7]                      # 형태: [B, N, 4] (index 3,4,5,6)
        
        # 좌표 피처와 도구 정보를 하나의 벡터로 이어붙임
        mlp_input = torch.cat([encoded_pos, tool_cond], dim=-1) # 형태: [B, N, 132]

        # [Step 4] 변위(u) 추론 경로
        u_offsets = [0] + torch.cumsum(torch.tensor(self.u_sizes), dim=0).tolist()
        x_u = mlp_input # 🌟 encoded_pos 대신 mlp_input 사용
        for i in range(len(self.u_layers) - 1):
            p = params_u_all[:, u_offsets[i]:u_offsets[i+1]]
            x_u = torch.tanh(self.u_layers[i](x_u, p))
        u_pred = self.u_layers[-1](x_u, params_u_all[:, u_offsets[-2]:u_offsets[-1]])

        # [Step 5] 탄성도(mu) 추론 경로
        mu_offsets = [0] + torch.cumsum(torch.tensor(self.mu_sizes), dim=0).tolist()
        x_mu = mlp_input # 🌟 encoded_pos 대신 mlp_input 사용
        for i in range(len(self.mu_layers) - 1):
            p = params_mu_all[:, mu_offsets[i]:mu_offsets[i+1]]
            x_mu = torch.tanh(self.mu_layers[i](x_mu, p))
        mu_raw = self.mu_layers[-1](x_mu, params_mu_all[:, mu_offsets[-2]:mu_offsets[-1]])
        
        # [Step 6] 물리적 하한선 적용
        mu_pred = 0.1 + torch.sigmoid(mu_raw) * 10.0 
        
        return u_pred, mu_pred