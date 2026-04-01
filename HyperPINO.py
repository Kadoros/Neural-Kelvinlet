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
        # 1. 공통 피처 추출기 (하이퍼네트워크의 뿌리)
        self.hyper_net = PointNetfeat(global_feat=True)
        
        # 2. 푸리에 인코딩 설정 (고주파 정보 활성화)
        # input_dim(3) -> target_width // 2 로 맵핑
        self.register_buffer("FF", torch.randn(3, target_width // 2) * freq)
        encoded_dim = target_width # sin, cos 합쳐서 target_width 차원

        # --------------------------------------------------
        # 3. 변위(u) 전용 타겟 네트워크 및 프로젝션
        # --------------------------------------------------
        self.u_layers = nn.ModuleList([
            FCLayer_batch(encoded_dim, target_width),
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 3) # u_x, u_y, u_z
        ])
        self.u_sizes = [l.get_param_size() for l in self.u_layers]
        self.param_proj_u = nn.Linear(1024, sum(self.u_sizes))

        # --------------------------------------------------
        # 4. 탄성도(mu) 전용 타겟 네트워크 및 프로젝션
        # --------------------------------------------------
        self.mu_layers = nn.ModuleList([
            FCLayer_batch(encoded_dim, target_width),
            FCLayer_batch(target_width, target_width),
            FCLayer_batch(target_width, 1) # mu_raw
        ])
        self.mu_sizes = [l.get_param_size() for l in self.mu_layers]
        self.param_proj_mu = nn.Linear(1024, sum(self.mu_sizes))

        # 5. 초기화 전략 (선배님 스타일 + 질문자님 참교육)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            # u 프로젝션 초기화
            torch.nn.init.uniform_(self.param_proj_u.weight, -1e-3, 1e-3)
            # mu 프로젝션 초기화 및 하한선 설정
            torch.nn.init.uniform_(self.param_proj_mu.weight, -1e-3, 1e-3)
            # mu의 마지막 바이어스를 조정하여 초기 mu_pred가 약 0.07이 되게 함
            self.param_proj_mu.bias.data[-1] = -2.63

    def input_encoding(self, pos):
        """좌표를 고주파 영역으로 맵핑하는 푸리에 인코딩"""
        x_proj = torch.matmul(pos, self.FF)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, pos, full_input):
        # [Step 1] 전역 피처 추출
        global_feat = self.hyper_net(full_input)
        
        # [Step 2] 각 타겟 네트워크를 위한 가중치 분리 생성
        params_u_all = self.param_proj_u(global_feat)
        params_mu_all = self.param_proj_mu(global_feat)
        
        # [Step 3] 푸리에 인코딩 적용 (pos -> encoded_pos)
        encoded_pos = self.input_encoding(pos)

        # [Step 4] 변위(u) 추론 경로
        u_offsets = [0] + torch.cumsum(torch.tensor(self.u_sizes), dim=0).tolist()
        x_u = encoded_pos
        for i in range(len(self.u_layers) - 1):
            p = params_u_all[:, u_offsets[i]:u_offsets[i+1]]
            x_u = torch.tanh(self.u_layers[i](x_u, p))
        u_pred = self.u_layers[-1](x_u, params_u_all[:, u_offsets[-2]:u_offsets[-1]])

        # [Step 5] 탄성도(mu) 추론 경로
        mu_offsets = [0] + torch.cumsum(torch.tensor(self.mu_sizes), dim=0).tolist()
        x_mu = encoded_pos
        for i in range(len(self.mu_layers) - 1):
            p = params_mu_all[:, mu_offsets[i]:mu_offsets[i+1]]
            x_mu = torch.tanh(self.mu_layers[i](x_mu, p))
        mu_raw = self.mu_layers[-1](x_mu, params_mu_all[:, mu_offsets[-2]:mu_offsets[-1]])
        
        # [Step 6] 물리적 하한선 적용 (질문자님 스타일)
        mu_pred = 0.03 + torch.sigmoid(mu_raw) * 10.0 
        
        return u_pred, mu_pred