import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 128]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())
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
        self.branch = SimpleMLP(branch_in, latent_dim * out_channels)
        self.trunk = SimpleMLP(trunk_in, latent_dim * out_channels)

    def forward(self, cond, pos):
        B, N = cond.shape[0], pos.shape[1]
        b_out = self.branch(cond).view(B, 1, self.out_channels, self.latent_dim)
        t_out = self.trunk(pos).view(B, N, self.out_channels, self.latent_dim)
        return torch.sum(b_out * t_out, dim=-1)

class SimplePINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_u = DeepONet(branch_in=4, trunk_in=3, out_channels=3)
        self.net_mu = DeepONet(branch_in=4, trunk_in=3, out_channels=1)

    def forward(self, tool_action, coords):
        u_pred = self.net_u(tool_action, coords)
    
        mu_raw = self.net_mu(tool_action, coords)
        # mu = exp(log_mu) 방식으로 예측. 
        # 모델이 -1.0을 뱉으면 mu는 0.36, -3.0을 뱉으면 0.05가 됨.
        # 자연스럽게 양수 범위를 탐색하며 0으로 완전히 죽지 않음.
        mu_pred = torch.exp(mu_raw) 
        return u_pred, mu_pred