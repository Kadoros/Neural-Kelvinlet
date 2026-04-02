import torch
import torch.nn as nn
import numpy as np

# SIREN(Sin 활성화 함수)을 제대로 구현한 MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 128], is_trunk=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.is_trunk = is_trunk
        
        curr_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(curr_dim, h_dim))
            curr_dim = h_dim
        self.output_layer = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 첫 레이어는 30배 스케일링(SIREN 표준), 나머지는 1배
            w0 = 30.0 if (i == 0 and self.is_trunk) else 1.0
            x = torch.sin(w0 * x)
        return self.output_layer(x)

class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in, latent_dim=64, out_channels=3, is_trunk=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.branch = SimpleMLP(branch_in, latent_dim * out_channels, is_trunk=False)
        self.trunk = SimpleMLP(trunk_in, latent_dim * out_channels, is_trunk=is_trunk)

    def forward(self, cond, pos):
        B, N = cond.shape[0], pos.shape[1]
        b_out = self.branch(cond).view(B, 1, self.out_channels, self.latent_dim)
        t_out = self.trunk(pos).view(B, N, self.out_channels, self.latent_dim)
        return torch.sum(b_out * t_out, dim=-1)

class SimplePINO(nn.Module):
    def __init__(self, enc_dim=15): # 1.5 벽을 깨는 돋보기 차원
        super().__init__()
        self.enc_dim = enc_dim
        # Trunk 입력: 원본(3) + Fourier(3 * 15 * 2) = 93차원
        trunk_in_dim = 3 + (3 * enc_dim * 2)
        
        self.net_u = DeepONet(branch_in=4, trunk_in=trunk_in_dim, out_channels=3, is_trunk=True)
        self.net_mu = DeepONet(branch_in=4, trunk_in=trunk_in_dim, out_channels=1, is_trunk=True)

        # 주파수 밴드 설정 (NeRF 방식)
        freq_bands = torch.pow(2, torch.linspace(0, 10, enc_dim)) 
        self.register_buffer("freq_bands", freq_bands)

    def fourier_encode(self, x):
        # x: (B, N, 3) -> 93차원으로 뻥튀기하여 고해상도 정보 획득
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * np.pi))
            out.append(torch.cos(x * freq * np.pi))
        return torch.cat(out, dim=-1)

    def forward(self, tool_action, coords):
        # 1. 돋보기 씌우기
        coords_enc = self.fourier_encode(coords)
        # 2. 예측
        u_pred = self.net_u(tool_action, coords_enc)
        mu_raw = self.net_mu(tool_action, coords_enc)
        # 3. 0.01 하한선 유지
        mu_pred = torch.nn.functional.softplus(mu_raw) + 0.01
        return u_pred, mu_pred