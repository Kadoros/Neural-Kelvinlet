import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 128]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())  # PDE 계산을 위해 미분 가능한 Tanh 사용
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in, latent_dim=64, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Branch: 툴 액션(4ch) 처리
        self.branch = SimpleMLP(branch_in, latent_dim * out_channels)
        # Trunk: 좌표(3ch) 처리
        self.trunk = SimpleMLP(trunk_in, latent_dim * out_channels)

    def forward(self, cond, pos):
        B = cond.shape[0]
        N = pos.shape[1]

        # Branch output: (B, out, latent)
        # Trunk output: (B, N, out, latent)
        b_out = self.branch(cond).view(B, 1, self.out_channels, self.latent_dim)
        t_out = self.trunk(pos).view(B, N, self.out_channels, self.latent_dim)

        # 채널별 내적을 통한 결과 생성
        return torch.sum(b_out * t_out, dim=-1)

class SimplePINO(nn.Module):
    def __init__(self):
        super().__init__()
        # 변위(u) 예측: 3채널 출력
        self.net_u = DeepONet(branch_in=4, trunk_in=3, out_channels=3)
        # 강성(mu) 예측: 1채널 출력
        self.net_mu = DeepONet(branch_in=4, trunk_in=3, out_channels=1)

    def forward(self, tool_action, coords):
        u_pred = self.net_u(tool_action, coords)
        mu_pred = self.net_mu(tool_action, coords)
        mu_pred = torch.nn.functional.softplus(mu_raw) + 0.01
        return u_pred, mu_pred