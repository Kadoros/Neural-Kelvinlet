import torch
import torch.nn as nn


# 1. 가장 기초가 되는 MLP 클래스 (푸리에 변환 제거)
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


# 2. DeepONet 구조 (Branch: 조건 입력, Trunk: 좌표 입력)
class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in, latent_dim=64, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Branch: Tool Action (4ch)를 처리
        self.branch = SimpleMLP(branch_in, latent_dim * out_channels)
        # Trunk: Coordinates (3ch)를 처리
        self.trunk = SimpleMLP(trunk_in, latent_dim * out_channels)

    def forward(self, cond, pos):
        # cond: (Batch, 4), pos: (Batch, N, 3)
        B = cond.shape[0]
        N = pos.shape[1]

        B_out = self.branch(cond)  # (B, latent_dim * out_channels)
        T_out = self.trunk(pos)  # (B, N, latent_dim * out_channels)

        # Reshape for multi-channel output
        B_out = B_out.view(B, 1, self.out_channels, self.latent_dim)
        T_out = T_out.view(B, N, self.out_channels, self.latent_dim)

        # 내적(Dot product)을 통한 최종 예측
        out = torch.sum(B_out * T_out, dim=-1)  # (B, N, out_channels)
        return out


# 3. 최종 통합 모델 (u와 mu를 완전히 분리)
class SimplePINO(nn.Module):
    def __init__(self):
        super().__init__()
        # u 예측용 모델 (출력 3채널: dx, dy, dz)
        self.net_u = DeepONet(branch_in=4, trunk_in=3, out_channels=3)
        # mu 예측용 모델 (출력 1채널: stiffness)
        self.net_mu = DeepONet(branch_in=4, trunk_in=3, out_channels=1)

    def forward(self, tool_action, coords):
        # tool_action: (B, 4), coords: (B, N, 3)
        u_pred = self.net_u(tool_action, coords)
        mu_pred = self.net_mu(tool_action, coords)
        return u_pred, mu_pred


# --- 학습 루프 예시 ---


def train_step(model, tool_action, coords, u_gt, optimizer):
    optimizer.zero_grad()

    # 좌표에 미분 설정 (PDE 계산용)
    coords.requires_grad_(True)

    # 모델 예측
    u_pred, mu_pred = model(tool_action, coords)

    # 1. Data-driven Loss (u는 정답과 비교)
    loss_u = nn.MSELoss()(u_pred, u_gt)

    # 2. Physics-driven Loss (mu는 PDE로 학습)
    # 사용자님이 작성하신 compute_static_pde_loss 함수 호출
    loss_pde = compute_static_pde_loss(coords, u_pred, mu_pred)

    # 전체 손실 합산
    total_loss = loss_u + 0.01 * loss_pde  # 계수는 튜닝 필요
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
