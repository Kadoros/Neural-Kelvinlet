import torch
import torch.nn as nn
import numpy as np

# 1. 활성화 함수를 Sin으로 변경 (PDE 2차 미분 신호를 살리는 핵심)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(curr_dim, h_dim))
            curr_dim = h_dim
        self.output_layer = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            # Tanh 대신 Sin 사용 (SIREN 구조) -> u의 미분값이 정교해짐
            x = torch.sin(layer(x)) 
        return self.output_layer(x)

# 2. 기존 DeepONet 구조 유지
class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in, latent_dim=64, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.branch = SimpleMLP(branch_in, latent_dim * out_channels)
        self.trunk = SimpleMLP(trunk_in, latent_dim * out_channels)

    def forward(self, cond, pos):
        B, N = cond.shape[0], pos.shape[1]
        b_out = self.branch(cond).view(B, 1, self.out_channels, self.latent_dim)
        t_out = self.trunk(pos).view(B, N, self.out_channels, self.latent_dim)
        return torch.sum(b_out * t_out, dim=-1)

# 3. 퓨리에 인코딩이 통합된 최종 모델
class SimplePINO(nn.Module):
    def __init__(self, enc_dim=15): # enc_dim은 해상도 (10~20 추천)
        super().__init__()
        self.enc_dim = enc_dim
        
        # Trunk 입력 차원: x,y,z(3) + sin/cos 세트(3 * enc_dim * 2) = 3 + 90 = 93
        trunk_in_dim = 3 + (3 * enc_dim * 2)
        
        # Branch는 툴 액션(4) 그대로, Trunk는 고해상도 좌표(93) 입력
        self.net_u = DeepONet(branch_in=4, trunk_in=trunk_in_dim, out_channels=3)
        self.net_mu = DeepONet(branch_in=4, trunk_in=trunk_in_dim, out_channels=1)

        # 돋보기(Fourier) 역할을 할 주파수 밴드 설정
        # 모델이 아주 세밀한 굴곡(High-frequency)을 보게 해줌
        freq_bands = torch.pow(2, torch.linspace(0, 6, enc_dim)) # 2^0 ~ 2^6
        self.register_buffer("freq_bands", freq_bands)

    def fourier_encode(self, x):
        # x: (B, N, 3)
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * np.pi))
            out.append(torch.cos(x * freq * np.pi))
        return torch.cat(out, dim=-1) # (B, N, 93)

    def forward(self, tool_action, coords):
        # 1. 좌표를 Fourier Feature로 변환 (해상도 뻥튀기)
        coords_enc = self.fourier_encode(coords)
        
        # 2. 딥오넷 구조 그대로 사용하여 u와 mu 예측
        u_pred = self.net_u(tool_action, coords_enc)
        mu_raw = self.net_mu(tool_action, coords_enc)
    
        # 3. mu는 softplus + 0.01로 물리적 하한선 강제
        mu_pred = torch.nn.functional.softplus(mu_raw) + 0.01
        return u_pred, mu_pred